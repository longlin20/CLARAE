"""
DILATE Loss Package

Official implementation from:
    Le Guen, V., & Thome, N. (2019).
    "Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models"
    NeurIPS 2019
    https://github.com/vincent-leguen/DILATE

This package provides:
- soft_dtw: Soft Dynamic Time Warping (differentiable DTW)
- path_soft_dtw: DTW path computation for temporal alignment
- dilate_loss: Combined shape and temporal distortion loss
- soft_dtw_loss: Standalone Soft-DTW wrapper using pysdtw library
"""

from .soft_dtw_loss import SoftDTWLoss

__all__ = [
    'SoftDTWLoss',
]
