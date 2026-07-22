"""
R² and MSE Metrics for EGM Signal Reconstruction
=================================================

R² = 1 - SS_res / SS_tot (coefficient of determination)
MSE = mean((y_true - y_pred)²)
"""

import numpy as np
import torch


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Calculate R² for numpy arrays. Returns nan if signal has no variance."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < eps:
        return float('nan')
    return float(1 - ss_res / ss_tot)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MSE for numpy arrays."""
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Calculate R² for PyTorch tensors. Returns nan if signal has no variance."""
    with torch.no_grad():
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        if ss_tot < eps:
            return torch.tensor(float('nan'))
        return 1 - ss_res / ss_tot


def mse_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calculate MSE for PyTorch tensors."""
    with torch.no_grad():
        return torch.mean((y_true - y_pred) ** 2)


def calculate_mse_batch(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """
    Calculate per-signal MSE between original and reconstructed signals.

    Args:
        original: 2D array (N, L) of original signals
        reconstructed: 2D array (N, L) of reconstructed signals

    Returns:
        dict with 'mean' and 'std' of MSE values across signals
    """
    mse_values = np.mean((original - reconstructed) ** 2, axis=1)
    return {'mean': float(np.mean(mse_values)), 'std': float(np.std(mse_values))}
