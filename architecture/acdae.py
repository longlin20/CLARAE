import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ECAModule(nn.Module):

    def __init__(self, channels):
        super(ECAModule, self).__init__()
        self.channels = channels

        # Calcular kernel size según ecuación (7) del paper
        # k = |log2(C)/2 + 1/2|_odd
        self.kernel_size = self._calculate_kernel_size(channels)

        # 1D convolution for cross-channel interaction
        self.conv1d = nn.Conv1d(1, 1, kernel_size=self.kernel_size,
                                padding=self.kernel_size // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def _calculate_kernel_size(self, channels):
        """Calcular kernel size según ecuación (7) del paper"""
        k = int(abs((math.log2(channels) / 2) + (1 / 2)))
        # Asegurar que sea impar
        if k % 2 == 0:
            k += 1
        return k

    def forward(self, x):
        """
        Forward pass del ECA module
        x: input tensor (batch, channels, length)
        """
        # Global Average Pooling - Ecuación (5) del paper
        # gn(F) = 1/L * sum(fi) for i=1 to L
        gap = F.adaptive_avg_pool1d(x, 1)  # (batch, channels, 1)

        # Reshape para 1D convolution
        gap = gap.view(gap.size(0), 1, gap.size(1))  # (batch, 1, channels)

        # 1D convolution para cross-channel interaction - Ecuación (6)
        # ω = σ(Conv1d_k(gn(F)))
        weights = self.conv1d(gap)  # (batch, 1, channels)
        weights = self.sigmoid(weights)  # Apply sigmoid activation

        # Reshape back
        weights = weights.view(weights.size(0), weights.size(2), 1)  # (batch, channels, 1)

        # Apply attention weights
        return x * weights


class ACDAE(nn.Module):
    """
    ACDAE Implementation - Exacta según el paper IEEE 2022

    Siguiendo la Tabla I y descripción del paper:
    - 4 convolutional layers en encoder
    - 4 transposed convolutional layers en decoder
    - ECA modules solo en decoder (Trans_Conv_1 a Trans_Conv_3)
    - Skip connections simétricas
    """

    def __init__(self, input_channels=1, signal_size=1250):
        super(ACDAE, self).__init__()

        self.signal_size = signal_size

        # ==================== ENCODER ====================
        # Conv_1: 13x1 kernel, 16 filters, output: 600x16
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=13,
                               stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(16)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Conv_2: 7x1 kernel, 32 filters, output: 300x32
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7,
                               stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(32)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Conv_3: 7x1 kernel, 64 filters, output: 150x64
        self.conv3 = nn.Conv1d(32, 64, kernel_size=7,
                               stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(64)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Conv_4: 7x1 kernel, 128 filters, output: 75x128
        self.conv4 = nn.Conv1d(64, 128, kernel_size=7,
                               stride=1, padding='same')
        self.bn4 = nn.BatchNorm1d(128)
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.2)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ==================== DECODER ====================
        # Trans_Conv_1: 7x1 kernel, 128 filters, output: 75x128
        self.trans_conv1 = nn.ConvTranspose1d(128, 128, kernel_size=7,
                                              stride=1, padding=3)
        self.bn_trans1 = nn.BatchNorm1d(128)
        self.leaky_relu_trans1 = nn.LeakyReLU(negative_slope=0.2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        # ECA Module después de Trans_Conv_1
        self.eca1 = ECAModule(128)

        # Trans_Conv_2: 7x1 kernel, 64 filters, output: 150x64
        self.trans_conv2 = nn.ConvTranspose1d(128, 64, kernel_size=7,
                                              stride=1, padding=3)
        self.bn_trans2 = nn.BatchNorm1d(64)
        self.leaky_relu_trans2 = nn.LeakyReLU(negative_slope=0.2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        # ECA Module después de Trans_Conv_2
        self.eca2 = ECAModule(64)

        # Trans_Conv_3: 7x1 kernel, 32 filters, output: 300x32
        self.trans_conv3 = nn.ConvTranspose1d(64, 32, kernel_size=7,
                                              stride=1, padding=3)
        self.bn_trans3 = nn.BatchNorm1d(32)
        self.leaky_relu_trans3 = nn.LeakyReLU(negative_slope=0.2)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')

        # ECA Module después de Trans_Conv_3
        self.eca3 = ECAModule(32)

        # Trans_Conv_4: 13x1 kernel, 16 filters, output: 600x16
        self.trans_conv4 = nn.ConvTranspose1d(32, 16, kernel_size=13,
                                              stride=1, padding=6)
        self.bn_trans4 = nn.BatchNorm1d(16)
        self.leaky_relu_trans4 = nn.LeakyReLU(negative_slope=0.2)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')

        # Dense Layer: output 600x1
        self.final_conv = nn.Conv1d(16, 1, kernel_size=1)

    def encode(self, x):
        """
        Extraer características del bottleneck para clasificación
        Según el paper: bottleneck está en Conv_4 output (75x128)
        """
        # Encoder forward pass
        x = self.leaky_relu1(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = self.leaky_relu2(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)

        x = self.leaky_relu3(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)

        x = self.leaky_relu4(self.bn4(self.conv4(x)))
        bottleneck = self.maxpool4(x)  # Este es el bottleneck (75x128)

        # Flatten para clasificación
        return bottleneck.view(bottleneck.size(0), -1)

    def forward(self, x):
        """
        Forward pass completo del ACDAE
        """
        # ==================== ENCODER ====================
        enc1 = self.leaky_relu1(self.bn1(self.conv1(x)))
        enc1_pool = self.maxpool1(enc1)

        enc2 = self.leaky_relu2(self.bn2(self.conv2(enc1_pool)))
        enc2_pool = self.maxpool2(enc2)

        enc3 = self.leaky_relu3(self.bn3(self.conv3(enc2_pool)))
        enc3_pool = self.maxpool3(enc3)

        enc4 = self.leaky_relu4(self.bn4(self.conv4(enc3_pool)))
        enc4_pool = self.maxpool4(enc4)  # Bottleneck

        # ==================== DECODER ====================
        # Trans_Conv_1 + ECA
        dec1 = self.leaky_relu_trans1(self.bn_trans1(self.trans_conv1(enc4_pool)))
        dec1_up = self.upsample1(dec1)
        dec1_eca = self.eca1(dec1_up)

        # Skip connection: addition con enc4
        dec1_skip = self._match_and_add(dec1_eca, enc4)

        # Trans_Conv_2 + ECA
        dec2 = self.leaky_relu_trans2(self.bn_trans2(self.trans_conv2(dec1_skip)))
        dec2_up = self.upsample2(dec2)
        dec2_eca = self.eca2(dec2_up)

        # Skip connection: addition con enc3
        dec2_skip = self._match_and_add(dec2_eca, enc3)

        # Trans_Conv_3 + ECA
        dec3 = self.leaky_relu_trans3(self.bn_trans3(self.trans_conv3(dec2_skip)))
        dec3_up = self.upsample3(dec3)
        dec3_eca = self.eca3(dec3_up)

        # Skip connection: addition con enc2
        dec3_skip = self._match_and_add(dec3_eca, enc2)

        # Trans_Conv_4 (sin ECA)
        dec4 = self.leaky_relu_trans4(self.bn_trans4(self.trans_conv4(dec3_skip)))
        dec4_up = self.upsample4(dec4)

        # Skip connection: addition con enc1
        dec4_skip = self._match_and_add(dec4_up, enc1)

        # Dense Layer (final output)
        output = self.final_conv(dec4_skip)

        # Ajustar tamaño final si es necesario
        if output.size(-1) != self.signal_size:
            output = F.interpolate(output, size=self.signal_size,
                                   mode='linear', align_corners=False)

        return output

    def _match_and_add(self, decoder_tensor, encoder_tensor):
        if decoder_tensor.size(-1) != encoder_tensor.size(-1):
            # Interpolar el tensor más pequeño
            target_size = max(decoder_tensor.size(-1), encoder_tensor.size(-1))

            if decoder_tensor.size(-1) < target_size:
                decoder_tensor = F.interpolate(decoder_tensor, size=target_size,
                                               mode='linear', align_corners=False)
            if encoder_tensor.size(-1) < target_size:
                encoder_tensor = F.interpolate(encoder_tensor, size=target_size,
                                               mode='linear', align_corners=False)

        return decoder_tensor + encoder_tensor