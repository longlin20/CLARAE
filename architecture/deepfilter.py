import torch
import torch.nn as nn
import torch.nn.functional as F


class LANLFilterModule(nn.Module):
    """
    LANL Filter Module with multiple kernel sizes and both linear and ReLU activations.
    """

    def __init__(self, in_channels, layers):
        super(LANLFilterModule, self).__init__()

        # Linear branches
        self.linear_conv3 = nn.Conv1d(in_channels, layers // 8, kernel_size=3, padding='same')
        self.linear_conv5 = nn.Conv1d(in_channels, layers // 8, kernel_size=5, padding='same')
        self.linear_conv9 = nn.Conv1d(in_channels, layers // 8, kernel_size=9, padding='same')
        self.linear_conv15 = nn.Conv1d(in_channels, layers // 8, kernel_size=15, padding='same')

        # Non-linear (ReLU) branches
        self.nonlinear_conv3 = nn.Conv1d(in_channels, layers // 8, kernel_size=3, padding='same')
        self.nonlinear_conv5 = nn.Conv1d(in_channels, layers // 8, kernel_size=5, padding='same')
        self.nonlinear_conv9 = nn.Conv1d(in_channels, layers // 8, kernel_size=9, padding='same')
        self.nonlinear_conv15 = nn.Conv1d(in_channels, layers // 8, kernel_size=15, padding='same')

    def forward(self, x):
        # Linear branches
        lb0 = self.linear_conv3(x)
        lb1 = self.linear_conv5(x)
        lb2 = self.linear_conv9(x)
        lb3 = self.linear_conv15(x)

        # Non-linear branches (with ReLU)
        nlb0 = F.relu(self.nonlinear_conv3(x))
        nlb1 = F.relu(self.nonlinear_conv5(x))
        nlb2 = F.relu(self.nonlinear_conv9(x))
        nlb3 = F.relu(self.nonlinear_conv15(x))

        # Concatenate all branches
        output = torch.cat([lb0, lb1, lb2, lb3, nlb0, nlb1, nlb2, nlb3], dim=1)

        return output


class LANLFilterModuleDilated(nn.Module):
    """
    LANL Filter Module with dilated convolutions.
    """

    def __init__(self, in_channels, layers):
        super(LANLFilterModuleDilated, self).__init__()

        # Linear branches with dilation
        self.linear_conv5 = nn.Conv1d(in_channels, layers // 6, kernel_size=5,
                                      padding='same', dilation=3)
        self.linear_conv9 = nn.Conv1d(in_channels, layers // 6, kernel_size=9,
                                      padding='same', dilation=3)
        self.linear_conv15 = nn.Conv1d(in_channels, layers // 6, kernel_size=15,
                                       padding='same', dilation=3)

        # Non-linear (ReLU) branches with dilation
        self.nonlinear_conv5 = nn.Conv1d(in_channels, layers // 6, kernel_size=5,
                                         padding='same', dilation=3)
        self.nonlinear_conv9 = nn.Conv1d(in_channels, layers // 6, kernel_size=9,
                                         padding='same', dilation=3)
        self.nonlinear_conv15 = nn.Conv1d(in_channels, layers // 6, kernel_size=15,
                                          padding='same', dilation=3)

    def forward(self, x):
        # Linear branches
        lb1 = self.linear_conv5(x)
        lb2 = self.linear_conv9(x)
        lb3 = self.linear_conv15(x)

        # Non-linear branches (with ReLU)
        nlb1 = F.relu(self.nonlinear_conv5(x))
        nlb2 = F.relu(self.nonlinear_conv9(x))
        nlb3 = F.relu(self.nonlinear_conv15(x))

        # Concatenate all branches
        output = torch.cat([lb1, lb2, lb3, nlb1, nlb2, nlb3], dim=1)

        return output


class DeepFilter(nn.Module):
    """
    DeepFilter model implementation based on LANL filter modules.

    This model uses alternating LANL filter modules (regular and dilated)
    with dropout and batch normalization for signal denoising.
    """

    def __init__(self, input_channels=1, signal_size=1250, dropout_rate=0.4):
        super(DeepFilter, self).__init__()

        self.signal_size = signal_size

        # First LANL module (64 filters) -> outputs 64 channels (8 * 8)
        self.lanl1 = LANLFilterModule(input_channels, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(64)  # 64 = 8 * (64//8) from LANLFilterModule

        # First dilated LANL module (60 filters to get 60 output channels)
        # 60 filters -> 6 * (60//6) = 6 * 10 = 60 channels
        self.lanl_dilated1 = LANLFilterModuleDilated(64, 60)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.bn2 = nn.BatchNorm1d(60)  # 60 = 6 * (60//6) from LANLFilterModuleDilated

        # Second LANL module (32 filters) -> outputs 32 channels (8 * 4)
        self.lanl2 = LANLFilterModule(60, 32)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.bn3 = nn.BatchNorm1d(32)

        # Second dilated LANL module (30 filters to get 30 output channels)
        # 30 filters -> 6 * (30//6) = 6 * 5 = 30 channels
        self.lanl_dilated2 = LANLFilterModuleDilated(32, 30)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.bn4 = nn.BatchNorm1d(30)  # 30 = 6 * (30//6) from LANLFilterModuleDilated

        # Third LANL module (16 filters) -> outputs 16 channels (8 * 2)
        self.lanl3 = LANLFilterModule(30, 16)
        self.dropout5 = nn.Dropout(dropout_rate)
        self.bn5 = nn.BatchNorm1d(16)

        # Third dilated LANL module (18 filters to get 18 output channels)
        # 18 filters -> 6 * (18//6) = 6 * 3 = 18 channels
        self.lanl_dilated3 = LANLFilterModuleDilated(16, 18)
        self.dropout6 = nn.Dropout(dropout_rate)
        self.bn6 = nn.BatchNorm1d(18)  # 18 = 6 * (18//6) from LANLFilterModuleDilated

        # Final prediction layer
        self.final_conv = nn.Conv1d(18, 1, kernel_size=9, padding='same')

    def forward(self, x):
        # Input shape: (batch_size, channels, sequence_length)

        # First block
        x = self.lanl1(x)
        x = self.dropout1(x)
        x = self.bn1(x)

        # First dilated block
        x = self.lanl_dilated1(x)
        x = self.dropout2(x)
        x = self.bn2(x)

        # Second block
        x = self.lanl2(x)
        x = self.dropout3(x)
        x = self.bn3(x)

        # Second dilated block
        x = self.lanl_dilated2(x)
        x = self.dropout4(x)
        x = self.bn4(x)

        # Third block
        x = self.lanl3(x)
        x = self.dropout5(x)
        x = self.bn5(x)

        # Third dilated block
        x = self.lanl_dilated3(x)
        x = self.dropout6(x)
        x = self.bn6(x)

        # Final prediction
        x = self.final_conv(x)

        return x