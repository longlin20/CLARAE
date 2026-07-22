"""
Implementation of DRNN approach presented in
Antczak, K. (2018). Deep recurrent neural networks for ECG signal denoising.
arXiv preprint arXiv:1807.11551.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class DRNN(nn.Module):
    def __init__(self, input_length=1250, hidden_size=64, dropout_rate=0.3):
        super(DRNN, self).__init__()

        # Store input_length for validation
        self.input_length = input_length

        # LSTM layer that processes the input signal
        # Remove dropout from LSTM (we'll add it after) to avoid warning
        self.lstm = nn.LSTM(
            input_size=1,  # Each time step has 1 feature (the signal value)
            hidden_size=hidden_size,  # Size of the hidden state
            batch_first=True,  # Input is [batch, seq_len, features]
            dropout=0.0,  # No dropout in LSTM itself
            bidirectional=False  # Original DRNN uses unidirectional LSTM
        )

        # Add dropout after LSTM
        self.dropout_lstm = nn.Dropout(dropout_rate)

        # Dense layers as shown in the DRNN architecture
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer to produce the denoised signal
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Input shape: (batch_size, channels, seq_length)
        batch_size, channels, seq_length = x.size()

        # LSTM expects: (batch_size, seq_length, features)
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, seq_length, channels)

        # Apply LSTM - get only the output, ignore hidden states for torchsummary compatibility
        lstm_out, _ = self.lstm(x)

        # Apply dropout after LSTM
        x = self.dropout_lstm(lstm_out)

        # Apply dense layers with activations to each time step
        x = self.dropout1(self.relu1(self.dense1(x)))
        x = self.dropout2(self.relu2(self.dense2(x)))
        x = self.output_layer(x)

        # Reshape back to (batch_size, channels, seq_length) for consistency with original code
        x = x.permute(0, 2, 1)

        return x
