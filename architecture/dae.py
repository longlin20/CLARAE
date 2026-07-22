import torch
import torch.nn as nn
import torch.nn.functional as F


class _UpsampleConv1DBlock(nn.Module):
    """
    Helper module for Upsampling followed by Conv1D and activation.
    This mimics the Conv1DTranspose2 behavior if it's implemented as Upsample + Conv.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride_upsample, activation_fn):
        super().__init__()
        self.stride_upsample = stride_upsample
        # Use 'nearest' mode for upsampling to be straightforward.
        # 'linear' could also be an option for 1D data.
        if self.stride_upsample > 1:
            self.upsample = nn.Upsample(scale_factor=stride_upsample, mode='nearest')

        # Calculate padding for Conv1D to maintain length (stride=1 for this Conv1D)
        # This assumes kernel_size is odd. If even, Keras 'same' can be more complex.
        # For K=16 (even), (16-1)//2 = 7. L_out = L_in for Conv1d stride 1.
        conv_padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=conv_padding)
        self.activation = activation_fn

    def forward(self, x):
        if self.stride_upsample > 1:
            x = self.upsample(x)
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class FCN_DAE(nn.Module):
    """
    PyTorch implementation of the FCN-DAE model.
    Chiang, H. T., et al. (2019). Noise reduction in ECG signals using fully
    convolutional denoising autoencoders.
    IEEE Access, 7, 60806-60813.
    """

    def __init__(self, input_length=1250, input_channels=1):
        super().__init__()

        elu = nn.ELU()
        kernel_size = 16

        # Encoder
        self.enc_conv1 = nn.Conv1d(input_channels, 40, kernel_size, stride=2, padding=(kernel_size - 1) // 2)
        self.enc_act1 = elu
        self.enc_bn1 = nn.BatchNorm1d(40)

        self.enc_conv2 = nn.Conv1d(40, 20, kernel_size, stride=2, padding=(kernel_size - 1) // 2)
        self.enc_act2 = elu
        self.enc_bn2 = nn.BatchNorm1d(20)

        self.enc_conv3 = nn.Conv1d(20, 20, kernel_size, stride=2, padding=(kernel_size - 1) // 2)
        self.enc_act3 = elu
        self.enc_bn3 = nn.BatchNorm1d(20)

        self.enc_conv4 = nn.Conv1d(20, 20, kernel_size, stride=2, padding=(kernel_size - 1) // 2)
        self.enc_act4 = elu
        self.enc_bn4 = nn.BatchNorm1d(20)

        self.enc_conv5 = nn.Conv1d(20, 40, kernel_size, stride=2, padding=(kernel_size - 1) // 2)
        self.enc_act5 = elu
        self.enc_bn5 = nn.BatchNorm1d(40)

        self.enc_conv6 = nn.Conv1d(40, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)  # Bottleneck
        self.enc_act6 = elu
        self.enc_bn6 = nn.BatchNorm1d(1)

        # Decoder (completamente corregido)
        self.dec_bn_pre_tconv1 = nn.BatchNorm1d(1)

        # Primera capa: 1→1, stride 1
        self.dec_tconv1 = _UpsampleConv1DBlock(1, 1, kernel_size, stride_upsample=1, activation_fn=elu)
        self.dec_bn1 = nn.BatchNorm1d(1)

        # Segunda capa: 1→40, stride 2
        self.dec_tconv2 = _UpsampleConv1DBlock(1, 40, kernel_size, stride_upsample=2, activation_fn=elu)
        self.dec_bn2 = nn.BatchNorm1d(40)

        # Tercera capa: 40→20, stride 2
        self.dec_tconv3 = _UpsampleConv1DBlock(40, 20, kernel_size, stride_upsample=2, activation_fn=elu)
        self.dec_bn3 = nn.BatchNorm1d(20)

        # Cuarta capa: 20→20, stride 2
        self.dec_tconv4 = _UpsampleConv1DBlock(20, 20, kernel_size, stride_upsample=2, activation_fn=elu)
        self.dec_bn4 = nn.BatchNorm1d(20)

        # Quinta capa: 20→20, stride 2
        self.dec_tconv5 = _UpsampleConv1DBlock(20, 20, kernel_size, stride_upsample=2, activation_fn=elu)
        self.dec_bn5 = nn.BatchNorm1d(20)

        # Sexta capa: 20→40, stride 2
        self.dec_tconv6 = _UpsampleConv1DBlock(20, 40, kernel_size, stride_upsample=2, activation_fn=elu)
        self.dec_bn6 = nn.BatchNorm1d(40)

        # Séptima capa (final): 40→1, stride 1
        self.dec_tconv7 = _UpsampleConv1DBlock(40, 1, kernel_size, stride_upsample=1, activation_fn=None)

    def encode(self, x):
        # Encoder hasta el bottleneck
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_act4(self.enc_bn4(self.enc_conv4(x)))
        x = self.enc_act5(self.enc_bn5(self.enc_conv5(x)))
        bottleneck = self.enc_act6(self.enc_bn6(self.enc_conv6(x)))  # Bottleneck (1 channel)

        # Flatten el bottleneck manteniendo su tamaño original
        # bottleneck shape: (batch, 1, length) -> (batch, length)
        latent = bottleneck.squeeze(1)  # Remove channel dimension

        return latent

    def forward(self, x):
        # Encoder
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_act4(self.enc_bn4(self.enc_conv4(x)))
        x = self.enc_act5(self.enc_bn5(self.enc_conv5(x)))
        x = self.enc_act6(self.enc_bn6(self.enc_conv6(x)))

        # Decoder
        x = self.dec_bn_pre_tconv1(x)
        x = self.dec_tconv1(x)
        x = self.dec_bn1(x)

        x = self.dec_tconv2(x)
        x = self.dec_bn2(x)

        x = self.dec_tconv3(x)
        x = self.dec_bn3(x)

        x = self.dec_tconv4(x)
        x = self.dec_bn4(x)

        x = self.dec_tconv5(x)
        x = self.dec_bn5(x)

        x = self.dec_tconv6(x)
        x = self.dec_bn6(x)

        x = self.dec_tconv7(x)

        # Ensure output has the same length as input using adaptive resizing if needed
        if x.size(2) != 1250:  # Default input_length
            x = F.interpolate(x, size=1250, mode='linear', align_corners=False)

        return x


class CNN_DAE(nn.Module):
    """
    PyTorch implementation of the CNN-DAE model (variant with Dense layers).
    """

    def __init__(self, input_length=1250, input_channels=1):
        super().__init__()
        self.input_length = input_length

        elu = nn.ELU()
        kernel_size = 16

        # Encoder (identical to FCN_DAE)
        self.enc_conv1 = nn.Conv1d(input_channels, 40, kernel_size, stride=2, padding=(kernel_size - 1) // 2)
        self.enc_act1 = elu
        self.enc_bn1 = nn.BatchNorm1d(40)

        self.enc_conv2 = nn.Conv1d(40, 20, kernel_size, stride=2, padding=(kernel_size - 1) // 2)
        self.enc_act2 = elu
        self.enc_bn2 = nn.BatchNorm1d(20)

        self.enc_conv3 = nn.Conv1d(20, 20, kernel_size, stride=2, padding=(kernel_size - 1) // 2)
        self.enc_act3 = elu
        self.enc_bn3 = nn.BatchNorm1d(20)

        self.enc_conv4 = nn.Conv1d(20, 20, kernel_size, stride=2, padding=(kernel_size - 1) // 2)
        self.enc_act4 = elu
        self.enc_bn4 = nn.BatchNorm1d(20)

        self.enc_conv5 = nn.Conv1d(20, 40, kernel_size, stride=2, padding=(kernel_size - 1) // 2)
        self.enc_act5 = elu
        self.enc_bn5 = nn.BatchNorm1d(40)

        self.enc_conv6 = nn.Conv1d(40, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)  # Bottleneck
        self.enc_act6 = elu
        self.enc_bn6 = nn.BatchNorm1d(1)

        # Decoder
        # Note: Keras code has BN after Conv1DTranspose2 blocks
        self.dec_tconv1 = _UpsampleConv1DBlock(1, 1, kernel_size, stride_upsample=1, activation_fn=elu)
        self.dec_bn1 = nn.BatchNorm1d(1)

        self.dec_tconv2 = _UpsampleConv1DBlock(1, 40, kernel_size, stride_upsample=2, activation_fn=elu)
        self.dec_bn2 = nn.BatchNorm1d(40)

        self.dec_tconv3 = _UpsampleConv1DBlock(40, 20, kernel_size, stride_upsample=2, activation_fn=elu)
        self.dec_bn3 = nn.BatchNorm1d(20)

        self.dec_tconv4 = _UpsampleConv1DBlock(20, 20, kernel_size, stride_upsample=2, activation_fn=elu)
        self.dec_bn4 = nn.BatchNorm1d(20)

        self.dec_tconv5 = _UpsampleConv1DBlock(20, 1, kernel_size, stride_upsample=2, activation_fn=elu)

        self.flatten = nn.Flatten(start_dim=1)  # Flatten channels and length

        # Calculamos el tamaño aproximado después de las convoluciones y upsampling
        # Encoder: input_length -> input_length/32
        # Decoder: input_length/32 -> input_length/2
        self.intermediate_length = input_length // 2

        self.dec_bn_after_flatten = nn.BatchNorm1d(self.intermediate_length)

        self.dec_dense1 = nn.Linear(self.intermediate_length, input_length // 2)
        self.dec_act_dense1 = elu
        self.dec_bn_dense1 = nn.BatchNorm1d(input_length // 2)
        self.dec_dropout = nn.Dropout(p=0.5)

        self.dec_dense2 = nn.Linear(input_length // 2, input_length)

    def encode(self, x):
        # Encoder hasta el bottleneck (misma lógica que FCN_DAE)
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_act4(self.enc_bn4(self.enc_conv4(x)))
        x = self.enc_act5(self.enc_bn5(self.enc_conv5(x)))
        bottleneck = self.enc_act6(self.enc_bn6(self.enc_conv6(x)))  # Bottleneck (1 channel)

        # Flatten el bottleneck manteniendo su tamaño original
        # bottleneck shape: (batch, 1, length) -> (batch, length)
        latent = bottleneck.squeeze(1)  # Remove channel dimension

        return latent

    def forward(self, x):
        batch_size = x.size(0)

        # Encoder
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_act4(self.enc_bn4(self.enc_conv4(x)))
        x = self.enc_act5(self.enc_bn5(self.enc_conv5(x)))
        x = self.enc_act6(self.enc_bn6(self.enc_conv6(x)))

        # Decoder - Parte convolucional
        x = self.dec_tconv1(x)
        x = self.dec_bn1(x)

        x = self.dec_tconv2(x)
        x = self.dec_bn2(x)

        x = self.dec_tconv3(x)
        x = self.dec_bn3(x)

        x = self.dec_tconv4(x)
        x = self.dec_bn4(x)

        x = self.dec_tconv5(x)

        # Flatten y capas densas
        x = self.flatten(x)

        # Asegurar que el tamaño es el esperado usando adaptive pooling si es necesario
        if x.size(1) != self.intermediate_length:
            x = x.view(batch_size, 1, -1)  # (batch_size, 1, flattened_length)
            x = F.adaptive_avg_pool1d(x, self.intermediate_length)
            x = x.view(batch_size, -1)  # (batch_size, intermediate_length)

        x = self.dec_bn_after_flatten(x)

        x = self.dec_dense1(x)
        x = self.dec_act_dense1(x)
        x = self.dec_bn_dense1(x)
        x = self.dec_dropout(x)

        x = self.dec_dense2(x)

        # Reshape para formato de salida: (batch_size, channels, length)
        x = x.view(batch_size, 1, self.input_length)

        return x
