"""Soft positive consistency for geometry-valid camera dropout."""

import torch
from torch import nn


class CameraDropConsistencyLoss(nn.Module):
    """Distil confident positive evidence without inventing background labels."""

    def __init__(self, confidence_threshold=0.3):
        super().__init__()
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0,1]")
        self.confidence_threshold = float(confidence_threshold)

    def forward(self, dropped_prediction, full_prediction):
        if dropped_prediction.shape != full_prediction.shape:
            raise ValueError("full and camera-dropped predictions must have equal shape")
        teacher = full_prediction.detach().clamp(0.0, 1.0)
        confident = teacher >= self.confidence_threshold
        if not confident.any():
            return dropped_prediction.sum() * 0.0
        # Stronger teacher responses carry more weight, while uncertain and
        # background locations have exactly zero consistency gradient.
        error = (dropped_prediction - teacher).pow(2)
        weight = teacher * confident.to(teacher.dtype)
        return (error * weight).sum() / weight.sum().clamp_min(1e-6)
