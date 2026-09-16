import torch
from torch import nn
import torch.nn.functional as F


class ConfuseGaussianMSE(nn.Module):
    """Soften supervision where a confident detection lacks an observed label.

    ``x`` is a model heatmap, ``target`` a sparse observed occupancy map, and
    ``kernel`` the dataset Gaussian kernel. The loss never reads hidden labels.
    """

    def __init__(self, confuse_pred_thr=0.3, beta=0.1, mirror=True):
        super().__init__()
        if not 0 <= confuse_pred_thr <= 1:
            raise ValueError('confuse_pred_thr must be in [0, 1]')
        if beta < 0:
            raise ValueError('beta must be nonnegative')
        self.confuse_pred_thr = confuse_pred_thr
        self.beta = beta
        self.mirror = mirror

    def _traget_transform(self, x, target, kernel):
        """Match output resolution and turn observed points into a soft target."""
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(
                target,
                kernel.float().to(target.device),
                padding=int((kernel.shape[-1] - 1) / 2),
            )
        return target.clamp(0, 1)

    def forward(self, x, target, kernel):
        """Average ordinary and confusion-aware squared errors over all pixels."""
        soft_gt = self._traget_transform(x, target, kernel)

        normal_sq = (x - soft_gt) ** 2
        if self.mirror:
            confuse_sq = (x - torch.ones_like(x)) ** 2
        else:
            confuse_sq = normal_sq

        # assignment must not backprop through the thresholding
        confuse_mask = (x.detach() >= self.confuse_pred_thr).float()

        confuse_term = (1 - soft_gt) * self.beta * confuse_sq + soft_gt * normal_sq
        blended = confuse_mask * confuse_term + (1 - confuse_mask) * normal_sq

        return blended.mean()
