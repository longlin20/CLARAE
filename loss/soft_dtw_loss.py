"""
SoftDTW Loss Wrapper using pysdtw library

This is a simplified wrapper around the pysdtw.SoftDTW implementation
for easy integration with PyTorch training loops.

The pysdtw library is much faster than the original soft_dtw.py implementation
because it uses optimized C++ code with numba acceleration.

Usage:
    from loss.soft_dtw_loss import SoftDTWLoss

    criterion = SoftDTWLoss(gamma=1.0, use_cuda=False)
    loss = criterion(pred, target)

Reference:
    - pysdtw library: https://github.com/toinsson/pysdtw
    - Original DTW: Sakoe & Chiba (1978)
    - Soft-DTW: Cuturi & Blondel (2017)
"""
import torch
import torch.nn as nn
from pysdtw import SoftDTW


class SoftDTWLoss(nn.Module):
    """
    Wrapper for SoftDTW that returns scalar loss (mean over batch)

    Args:
        gamma (float): Smoothing parameter for soft-DTW
                      - Smaller values (e.g., 0.01): More like hard DTW
                      - Larger values (e.g., 1.0): Smoother, more robust
        use_cuda (bool): Whether to use CUDA for DTW computation
                        Note: May have compatibility issues with newer Python versions

    Input Shape:
        - pred: [batch, time, channels] or [batch, channels, time]
        - target: [batch, time, channels] or [batch, channels, time]

    Output:
        - loss: Scalar tensor (mean DTW distance over batch)
    """

    def __init__(self, gamma=1.0, use_cuda=False):
        super().__init__()
        self.sdtw = SoftDTW(gamma=gamma, use_cuda=use_cuda)
        self.use_cuda = use_cuda

    def forward(self, pred, target):
        """
        Compute Soft-DTW loss

        Args:
            pred: Predicted time series
            target: Target time series

        Returns:
            Scalar loss (mean over batch)
        """
        # If not using CUDA for DTW but tensors are on CUDA, move to CPU temporarily
        original_device = pred.device
        if not self.use_cuda and pred.is_cuda:
            pred = pred.cpu()
            target = target.cpu()

        # SoftDTW returns [batch] tensor, we need scalar for backward
        loss = self.sdtw(pred, target).mean()

        # Move loss back to original device if needed
        if loss.device != original_device:
            loss = loss.to(original_device)

        return loss

    def __repr__(self):
        return f"SoftDTWLoss(gamma={self.sdtw.gamma}, use_cuda={self.use_cuda})"
