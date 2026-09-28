import torch
from torch import nn
import torch.nn.functional as F


class BRLGaussianMSE(nn.Module):
    """
    Heatmap adaptation of Background Recalibration Loss (BRL / Selective-IoU idea).

    Soft GT is built like GaussianMSE. With external pseudo labels, pixels are
    assigned with priority GT > pseudo > confuse > easy negative.  Pseudo loss
    is normalized separately and scaled by ``lambda_pseudo`` so its weight is
    meaningful even when pseudo pixels are sparse.

    On self-confuse pixels, either:
      - mirror=True: pull toward 1 (do not force background), scaled by beta
      - mirror=False: keep MSE-to-GT but down-weighted by beta
    """

    def __init__(self, pos_thr=0.1, confuse_pred_thr=0.3, beta=0.1,
                 mirror=True, use_confuse=True, pseudo_thr=0.1,
                 lambda_pseudo=0.1):
        super().__init__()
        self.pos_thr = pos_thr
        self.confuse_pred_thr = confuse_pred_thr
        self.beta = beta
        self.mirror = mirror
        self.use_confuse = use_confuse
        self.pseudo_thr = pseudo_thr
        self.lambda_pseudo = lambda_pseudo
        self.last_components = {}

    def forward(self, x, target, kernel, pseudo_target=None, pseudo_conf=None):
        """Compute the complete MVDet heatmap loss.

        Region priority is GT > external pseudo > self-confuse > easy negative.
        Passing no ``pseudo_target`` preserves the original BRL behaviour.  With
        ``use_confuse=False`` the remaining non-GT pixels use ordinary Gaussian
        MSE, which implements the pseudo-only experiment.
        """
        soft_gt = self._traget_transform(x, target, kernel)

        pos_mask = soft_gt >= self.pos_thr
        pseudo_mask = torch.zeros_like(pos_mask)
        soft_pseudo = None
        soft_pseudo_conf = None
        if pseudo_target is not None:
            soft_pseudo = self._traget_transform(x, pseudo_target, kernel)
            pseudo_mask = (soft_pseudo >= self.pseudo_thr) & ~pos_mask
            if pseudo_conf is None:
                pseudo_conf = (pseudo_target > 0).to(dtype=x.dtype)
            soft_pseudo_conf = self._traget_transform(x, pseudo_conf, kernel)
            soft_pseudo_conf = soft_pseudo_conf.clamp(min=0.0, max=1.0)

        rest_mask = ~(pos_mask | pseudo_mask)

        # assignment must not backprop through the thresholding
        if self.use_confuse:
            confuse_mask = rest_mask & (x.detach() >= self.confuse_pred_thr)
        else:
            confuse_mask = torch.zeros_like(rest_mask)
        easy_neg_mask = rest_mask & ~confuse_mask

        base_loss = x.new_zeros(())
        base_count = x.new_zeros(())

        if pos_mask.any():
            base_loss = base_loss + F.mse_loss(x[pos_mask], soft_gt[pos_mask], reduction="sum")
            base_count = base_count + pos_mask.sum()

        if easy_neg_mask.any():
            base_loss = base_loss + F.mse_loss(
                x[easy_neg_mask], soft_gt[easy_neg_mask], reduction="sum"
            )
            base_count = base_count + easy_neg_mask.sum()

        if confuse_mask.any():
            if self.mirror:
                mirror_tgt = torch.ones_like(x[confuse_mask])
                base_loss = base_loss + self.beta * F.mse_loss(
                    x[confuse_mask], mirror_tgt, reduction="sum"
                )
            else:
                base_loss = base_loss + self.beta * F.mse_loss(
                    x[confuse_mask], soft_gt[confuse_mask], reduction="sum"
                )
            base_count = base_count + confuse_mask.sum()

        base_loss = base_loss / base_count.clamp(min=1).float()

        pseudo_loss = x.new_zeros(())
        if pseudo_mask.any():
            weights = soft_pseudo_conf[pseudo_mask]
            squared_error = F.mse_loss(
                x[pseudo_mask], soft_pseudo[pseudo_mask], reduction="none"
            )
            pseudo_loss = (weights * squared_error).sum() / weights.sum().clamp(min=1e-6)

        total_loss = base_loss + self.lambda_pseudo * pseudo_loss
        self.last_components = {
            "base": base_loss.detach(),
            "pseudo": pseudo_loss.detach(),
            "gt_pixels": pos_mask.sum().detach(),
            "pseudo_pixels": pseudo_mask.sum().detach(),
            "confuse_pixels": confuse_mask.sum().detach(),
            "negative_pixels": easy_neg_mask.sum().detach(),
        }

        return total_loss

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(
                target,
                kernel.float().to(target.device),
                padding=int((kernel.shape[-1] - 1) / 2),
            )
        return target
