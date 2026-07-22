import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfONNLayer1D(nn.Module):
    """
    Self-Organized Operational Neural Network layer for 1D signals.
    Implements polynomial transformations with learnable weights.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding='same', dilation=1, use_bias=True, q=1):
        super(SelfONNLayer1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.q = q
        self.use_bias = use_bias

        # Calculate padding for 'same' mode
        if padding == 'same':
            self.padding = ((kernel_size - 1) * dilation) // 2
        else:
            self.padding = padding

        # Initialize weights for polynomial terms
        self.weights_onn = nn.Parameter(
            torch.zeros(q, out_channels, in_channels, kernel_size)
        )

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier uniform initialization
        for q_idx in range(self.q):
            nn.init.xavier_uniform_(self.weights_onn[q_idx])

        if self.bias is not None:
            bound = 0.01
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size = x.size(0)

        # Apply polynomial transformations: x^1, x^2, ..., x^q
        poly_terms = []
        for i in range(1, self.q + 1):
            poly_terms.append(torch.pow(x, i))

        # Concatenate polynomial terms along channel dimension
        x_poly = torch.cat(poly_terms, dim=1)  # (batch, q*in_channels, length)

        # Reshape weights for convolution
        # From (q, out_channels, in_channels, kernel_size)
        # to (out_channels, q*in_channels, kernel_size)
        w = self.weights_onn.view(self.q * self.in_channels, self.out_channels, self.kernel_size)
        w = w.permute(1, 0, 2)  # (out_channels, q*in_channels, kernel_size)

        # Perform 1D convolution
        out = F.conv1d(x_poly, w, bias=self.bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation)

        return out


class GatedSelfONN(nn.Module):
    """
    Gated Self-ONN block combining gating mechanism with polynomial operations.
    As described in paper Equation (5).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, q=1):
        super(GatedSelfONN, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.q = q

        # Gate and input branches using Self-ONN
        self.conv_gate = SelfONNLayer1D(
            in_channels, out_channels, kernel_size,
            stride=stride, padding='same', q=q
        )
        self.conv_input = SelfONNLayer1D(
            in_channels, out_channels, kernel_size,
            stride=stride, padding='same', q=q
        )

        # Dropout rate increases with q
        self.dropout = nn.Dropout(0.001 * q)

    def forward(self, x):
        # Gate branch with sigmoid activation
        x_g = self.conv_gate(x)
        x_g = torch.sigmoid(x_g)

        # Apply dropout if not a single channel
        if x_g.size(1) > 1:  # Check number of channels
            x_g = self.dropout(x_g)

        # Input branch (no activation)
        x_i = self.conv_input(x)

        # Element-wise multiplication - Equation (5): y = ONNi(x) ⊗ η(σ(ONNg(x)))
        return x_g * x_i


class GatedDeConv(nn.Module):
    """
    Gated Deconvolution (Transposed Convolution) block.
    As described in paper Equation (6).
    """

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=2):
        super(GatedDeConv, self).__init__()

        # Calculate padding for transpose convolution
        self.padding = (kernel_size - stride) // 2
        self.output_padding = stride - 1

        self.deconv_gate = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=self.padding,
            output_padding=self.output_padding
        )
        self.deconv_input = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=self.padding,
            output_padding=self.output_padding
        )
        self.dropout = nn.Dropout(0.001)

    def forward(self, x):
        # Equation (6): y = Deconvi(x) ⊗ η(σ(Deconvg(x)))
        x_g = torch.sigmoid(self.deconv_gate(x))

        # Apply dropout only if channels > 1
        if x_g.size(1) > 1:
            x_g = self.dropout(x_g)

        x_i = self.deconv_input(x)
        return x_g * x_i


class ChannelAttention(nn.Module):
    """
    Channel attention mechanism as described in the paper.
    Uses both max and average pooling with shared conv layer.
    Equations (7) and (8).
    """

    def __init__(self, channels, kernel_size=3):
        super(ChannelAttention, self).__init__()
        self.channels = channels

        # Shared convolutional layer for both avg and max pooled features
        self.shared_conv = nn.Conv1d(channels, channels, kernel_size,
                                     padding='same', groups=channels)
        self.dropout = nn.Dropout(0.001)

    def forward(self, x):
        # Global average pooling - shape: (batch, channels, 1)
        avg_pool = torch.mean(x, dim=2, keepdim=True)
        # Global max pooling - shape: (batch, channels, 1)
        max_pool = torch.max(x, dim=2, keepdim=True)[0]

        # Apply shared conv to each pooled feature
        avg_out = torch.sigmoid(self.shared_conv(avg_pool))
        max_out = torch.sigmoid(self.shared_conv(max_pool))

        # Add outputs to form attention mask - Equation (7)
        attention_mask = avg_out + max_out

        # Apply dropout only if channels > 1
        if self.channels > 1:
            attention_mask = self.dropout(attention_mask)

        # Apply attention - Equation (8): y = x ⊗ ω
        return x * attention_mask


class ResidualGate(nn.Module):
    """
    Residual Gate module as described in the paper.
    Equation (9).
    """

    def __init__(self, channels):
        super(ResidualGate, self).__init__()

        # 1x1 convolutions for input and residual paths
        self.conv_input = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_residual = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(0.1)  # Higher dropout on residual path

    def forward(self, x_input, x_residual):
        # Process input and residual with 1x1 convs
        input_features = self.conv_input(x_input)
        residual_features = self.conv_residual(x_residual)

        # Apply dropout to residual features for robustness
        residual_features = self.dropout(residual_features)

        # Generate gating mask - Equation (9)
        gate = torch.sigmoid(input_features + residual_features)

        # Apply gating to input
        return x_input * gate


class GatedONNEncoderBlock(nn.Module):
    """
    Encoder block with Gated Self-ONN, normalization, pooling, and attention.
    Following the exact structure shown in Figure 2.
    """

    def __init__(self, in_channels, out_channels, kernel_size=9, q=2):
        super(GatedONNEncoderBlock, self).__init__()

        # Gated Self-ONN convolution
        self.gated_onn = GatedSelfONN(in_channels, out_channels, kernel_size, stride=1, q=q)

        # Instance normalization (not group norm)
        self.norm = nn.InstanceNorm1d(out_channels, affine=True)

        # LeakyReLU activation
        self.activation = nn.LeakyReLU(0.2)

        # Max pooling
        self.maxpool = nn.MaxPool1d(2, stride=2)

        # Channel attention
        self.channel_attention = ChannelAttention(out_channels, kernel_size=3)

    def forward(self, x):
        # Gated Self-ONN
        x = self.gated_onn(x)

        # Instance Normalization
        x = self.norm(x)

        # LeakyReLU
        x = self.activation(x)

        # Max pooling
        x = self.maxpool(x)

        # Channel attention
        x = self.channel_attention(x)

        return x


class GatedDecoderBlock(nn.Module):
    """
    Decoder block with gated deconvolution, normalization, and attention.
    Following the exact structure shown in Figure 2.
    """

    def __init__(self, in_channels, out_channels, kernel_size=9):
        super(GatedDecoderBlock, self).__init__()

        # Gated deconvolution (stride=2 for upsampling)
        self.gated_deconv = GatedDeConv(in_channels, out_channels, kernel_size, stride=2)

        # Instance normalization
        self.norm = nn.InstanceNorm1d(out_channels, affine=True)

        # LeakyReLU activation
        self.activation = nn.LeakyReLU(0.2)

        # Channel attention
        self.channel_attention = ChannelAttention(out_channels, kernel_size=3)

    def forward(self, x):
        # Gated Deconv
        x = self.gated_deconv(x)

        # Instance Normalization
        x = self.norm(x)

        # LeakyReLU
        x = self.activation(x)

        # Channel attention
        x = self.channel_attention(x)

        return x


class FGDAE(nn.Module):
    """
    Fully Gated Denoising Autoencoder with Self-ONN layers.
    Following the exact architecture from the paper figure.
    """

    def __init__(self, signal_size=512, q=2):
        super(FGDAE, self).__init__()
        self.signal_size = signal_size
        self.q = q

        # Encoder path - following the paper's architecture exactly
        self.enc1 = GatedONNEncoderBlock(1, 16, kernel_size=9, q=q)  # 1 → 16
        self.enc2 = GatedONNEncoderBlock(16, 32, kernel_size=9, q=q)  # 16 → 32
        self.enc3 = GatedONNEncoderBlock(32, 64, kernel_size=9, q=q)  # 32 → 64
        self.enc4 = GatedONNEncoderBlock(64, 64, kernel_size=9, q=q)  # 64 → 64 (same)
        self.enc5 = GatedONNEncoderBlock(64, 1, kernel_size=9, q=q)  # 64 → 1

        # Decoder path - following the paper's architecture exactly
        self.dec1 = GatedDecoderBlock(1, 64, kernel_size=9)  # 1 → 64
        self.dec2 = GatedDecoderBlock(64, 64, kernel_size=9)  # 64 → 64 (same)
        self.dec3 = GatedDecoderBlock(64, 32, kernel_size=9)  # 64 → 32
        self.dec4 = GatedDecoderBlock(32, 16, kernel_size=9)  # 32 → 16
        self.dec5 = GatedDecoderBlock(16, 1, kernel_size=9)  # 16 → 1

        # Residual gates for skip connections
        self.residual_gate1 = ResidualGate(64)  # For enc4 → dec1
        self.residual_gate2 = ResidualGate(64)  # For enc3 → dec2
        self.residual_gate3 = ResidualGate(32)  # For enc2 → dec3
        self.residual_gate4 = ResidualGate(16)  # For enc1 → dec4

    def encode(self, x):
        """Extract bottleneck features for compatibility."""
        # Encoder forward pass
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)  # This is the bottleneck: 1 channel

        # Flatten bottleneck - for signal_size=512: 1×16=16, for 1250: 1×39=39
        return enc5.view(enc5.size(0), -1)

    def forward(self, x):
        # Encoder path with skip connections saved
        enc1 = self.enc1(x)  # 16 channels
        enc2 = self.enc2(enc1)  # 32 channels
        enc3 = self.enc3(enc2)  # 64 channels
        enc4 = self.enc4(enc3)  # 64 channels
        enc5 = self.enc5(enc4)  # 1 channel (bottleneck)

        # Decoder path with residual gated skip connections
        # First decoder block just uses bottleneck
        dec1 = self.dec1(enc5)  # 1 → 64 channels

        # Apply residual gate for skip connection from enc4
        enc4_matched = self._match_size(enc4, dec1)
        dec1 = self.residual_gate1(dec1, enc4_matched)

        dec2 = self.dec2(dec1)  # 64 → 64 channels
        enc3_matched = self._match_size(enc3, dec2)
        dec2 = self.residual_gate2(dec2, enc3_matched)

        dec3 = self.dec3(dec2)  # 64 → 32 channels
        enc2_matched = self._match_size(enc2, dec3)
        dec3 = self.residual_gate3(dec3, enc2_matched)

        dec4 = self.dec4(dec3)  # 32 → 16 channels
        enc1_matched = self._match_size(enc1, dec4)
        dec4 = self.residual_gate4(dec4, enc1_matched)

        output = self.dec5(dec4)  # 16 → 1 channel

        # Ensure output matches input size
        if output.size(2) != self.signal_size:
            output = F.interpolate(output, size=self.signal_size, mode='linear', align_corners=True)

        return output

    def _match_size(self, tensor1, tensor2):
        """Match tensor1 size to tensor2 size for skip connections."""
        if tensor1.size(2) != tensor2.size(2):
            # Interpolate tensor1 to match tensor2's size
            tensor1 = F.interpolate(tensor1, size=tensor2.size(2), mode='linear', align_corners=True)
        return tensor1

    @property
    def encoder(self):
        """Property to make FGDAE compatible with existing code."""

        class EncoderWrapper(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent

            def forward(self, x):
                return self.parent.encode(x)

        return EncoderWrapper(self)
