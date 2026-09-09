from torch import nn
import torch.nn.functional as F

from .heatmap import max_gaussian_target


class GaussianMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, target, kernel):
        target = self._traget_transform(x, target, kernel)
        return F.mse_loss(x, target)

    def _traget_transform(self, x, target, kernel):
        return max_gaussian_target(x, target, kernel)
