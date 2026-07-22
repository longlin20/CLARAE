import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder_CLARAE(nn.Module):
    """
    CLARAE Encoder — same filter progression as SC models (fi->fi//2->fi//4->fi//8),
    configurable dense_dim, returns only the latent vector (no skip connections).

    Architecture (fi=filters_initial):
        Layer 1: Conv(1      -> fi)     + BN + pool
        Layer 2: Conv(fi     -> fi//2)  + BN + pool
        Layer 3: Conv(fi//2  -> fi//4)  + BN + pool
        Layer 4: Conv(fi//4  -> fi//8)  + BN + pool
        FC: flatten -> dense_dim -> latent_dim  (tanh)
    """
    def __init__(self, input_channels, input_length, latent_dim,
                 filters_initial, dropout_rate=0.2, dense_dim=256):
        super().__init__()
        fi = filters_initial

        self.conv1 = nn.Conv1d(input_channels, fi,      kernel_size=7, padding='same')
        self.bn1   = nn.BatchNorm1d(fi)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(fi,      fi // 2, kernel_size=7, padding='same')
        self.bn2   = nn.BatchNorm1d(fi // 2)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(fi // 2, fi // 4, kernel_size=7, padding='same')
        self.bn3   = nn.BatchNorm1d(fi // 4)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(fi // 4, fi // 8, kernel_size=7, padding='same')
        self.bn4   = nn.BatchNorm1d(fi // 8)
        self.pool4 = nn.MaxPool1d(2)

        flatten_size = (input_length // 16) * (fi // 8)
        self.fc1     = nn.Linear(flatten_size, dense_dim)
        self.fc2     = nn.Linear(dense_dim, latent_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.3); x = self.pool1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.3); x = self.pool2(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.3); x = self.pool3(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.3); x = self.pool4(x)
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.dropout(self.fc1(x)), 0.3)
        return torch.tanh(self.fc2(x))


class Decoder_CLARAE(nn.Module):
    """
    CLARAE Decoder — symmetric mirror of Encoder_CLARAE.

    Architecture (fi=filters_initial):
        FC: latent_dim -> dense_dim -> (L//16)*(fi//8)  then reshape
        Layer 1: upsample + Conv(fi//8 -> fi//4) + BN
        Layer 2: upsample + Conv(fi//4 -> fi//2) + BN
        Layer 3: upsample + Conv(fi//2 -> fi)    + BN
        Layer 4: upsample + Conv(fi    -> 1)     + tanh
    """
    def __init__(self, input_length, latent_dim, filters_initial,
                 dropout_rate=0.2, dense_dim=256):
        super().__init__()
        fi = filters_initial
        self.input_length = input_length
        self.fi8 = fi // 8

        self.fc1 = nn.Linear(latent_dim, dense_dim)
        self.fc2 = nn.Linear(dense_dim, (input_length // 16) * (fi // 8))

        self.conv1    = nn.Conv1d(fi // 8, fi // 4, kernel_size=7, padding='same')
        self.bn1      = nn.BatchNorm1d(fi // 4)

        self.conv2    = nn.Conv1d(fi // 4, fi // 2, kernel_size=7, padding='same')
        self.bn2      = nn.BatchNorm1d(fi // 2)

        self.conv3    = nn.Conv1d(fi // 2, fi,      kernel_size=7, padding='same')
        self.bn3      = nn.BatchNorm1d(fi)

        self.conv_out = nn.Conv1d(fi, 1, kernel_size=7, padding='same')
        self.dropout  = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.leaky_relu(self.dropout(self.fc1(x)), 0.3)
        x = F.leaky_relu(self.fc2(x), 0.3)
        x = x.view(x.shape[0], self.fi8, -1)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.3)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.3)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.3)

        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if x.shape[-1] != self.input_length:
            x = F.interpolate(x, size=self.input_length, mode='linear', align_corners=True)
        return torch.tanh(self.conv_out(x))


class CLARAE(nn.Module):
    """
    CLARAE: Convolutional Latent Representation AutoEncoder

    Symmetric autoencoder for EGM signal processing.
    Filter progression mirrors SC models (fi->fi//2->fi//4->fi//8) with a
    configurable dense_dim FC bottleneck. No skip connections.

    Parameters:
    - input_channels:  1 for single-lead EGM
    - input_length:    signal length in samples (e.g. 1250)
    - latent_dim:      latent space dimension (e.g. 64)
    - filters_initial: base filter count (e.g. 64); channels go fi->fi//2->fi//4->fi//8
    - dropout_rate:    dropout (0.1-0.3)
    - dense_dim:       FC hidden size between flatten and latent (default 256)
    """
    def __init__(self, input_channels, input_length, latent_dim,
                 filters_initial, dropout_rate=0.2, dense_dim=256):
        super().__init__()
        self.encoder = Encoder_CLARAE(
            input_channels, input_length, latent_dim,
            filters_initial, dropout_rate, dense_dim,
        )
        self.decoder = Decoder_CLARAE(
            input_length, latent_dim, filters_initial, dropout_rate, dense_dim,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# =============================================================================
# CLARAE_SC: U-Net style architecture with skip connections (formerly CLARAE_V3)
# =============================================================================

class Encoder_CLARAE_SC(nn.Module):
    """
    CLARAE_V3 Encoder with skip connections

    Architecture (filters_initial=64): 1 → 64 → 64 → 32 → 32 → latent
    - Layer 1: 1 → filters_initial (pool)
    - Layer 2: filters_initial → filters_initial (pool)
    - Layer 3: filters_initial → filters_initial//2 (pool)
    - Layer 4: filters_initial//2 → filters_initial//2 (pool)
    - FC: flatten → 512 → latent
    """
    def __init__(self, input_channels, input_length, latent_dim, filters_initial, dropout_rate=0.2, dense_dim=256):
        super(Encoder_CLARAE_SC, self).__init__()
        self.filters_initial = filters_initial

        # Layer 1: input → filters_initial
        self.conv1 = nn.Conv1d(input_channels, filters_initial, kernel_size=7, padding="same")
        self.bn1 = nn.BatchNorm1d(filters_initial)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Layer 2: filters_initial → filters_initial
        self.conv2 = nn.Conv1d(filters_initial, filters_initial, kernel_size=7, padding="same")
        self.bn2 = nn.BatchNorm1d(filters_initial)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Layer 3: filters_initial → filters_initial // 2
        self.conv3 = nn.Conv1d(filters_initial, filters_initial // 2, kernel_size=7, padding="same")
        self.bn3 = nn.BatchNorm1d(filters_initial // 2)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Layer 4: filters_initial // 2 → filters_initial // 2
        self.conv4 = nn.Conv1d(filters_initial // 2, filters_initial // 2, kernel_size=7, padding="same")
        self.bn4 = nn.BatchNorm1d(filters_initial // 2)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        # Calculate flatten size (after 4 poolings: length / 16)
        flatten_size = (input_length // 16) * (filters_initial // 2)

        # FC layers: flatten → dense_dim → latent
        self.fc1 = nn.Linear(flatten_size, dense_dim)
        self.fc2 = nn.Linear(dense_dim, latent_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass with skip connection outputs

        Returns:
            latent: Latent vector
            skips: List of skip connection features [skip1, skip2, skip3, skip4]
        """
        skips = []

        # Layer 1: 1 → filters_initial
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        skips.append(x)
        x = self.pool1(x)

        # Layer 2: filters_initial → filters_initial
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        skips.append(x)
        x = self.pool2(x)

        # Layer 3: filters_initial → filters_initial // 2
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        skips.append(x)
        x = self.pool3(x)

        # Layer 4: filters_initial // 2 → filters_initial // 2
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        skips.append(x)
        x = self.pool4(x)

        # Flatten and FC layers
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(self.dropout(x), negative_slope=0.3)
        x = self.fc2(x)
        latent = torch.tanh(x)

        return latent, skips

