"""Mirror focal loss adapted to partially annotated MVDet heatmaps.

The original Background Recalibration Loss (BRL) operates on anchor labels.
MVDet instead predicts dense Gaussian heatmaps, so applying the anchor loss to
every background pixel would turn broad false-positive regions into positives.
This implementation only mirrors confident, spatially isolated local maxima,
caps their number using the known annotation propensity, and expands selected
maxima with the same Gaussian kernel used for the supervised target.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def partial_annotation_defaults(drop_percent):
    """Return conservative BRL defaults for a known annotation drop rate.

    ``beta`` follows the expected hidden/observed instance odds and is capped
    at one.  The confidence threshold decreases mildly when background labels
    are less trustworthy, but never goes below 0.60.
    """
    if not 0 <= drop_percent < 100:
        raise ValueError('drop_percent must be in [0, 100)')

    drop_rate = drop_percent / 100.0
    annotation_probability = 1.0 - drop_rate
    hidden_to_observed = drop_rate / max(annotation_probability, 1e-8)
    return {
        'drop_rate': drop_rate,
        'threshold': max(0.60, 0.75 - 0.25 * drop_rate),
        'beta': min(1.0, hidden_to_observed),
        'background_weight': annotation_probability,
    }


class MirrorFocalHeatmapLoss(nn.Module):
    """Hybrid Gaussian and mirror-focal loss for missing heatmap labels.

    Args:
        drop_rate: Fraction of instances omitted independently from each frame.
        threshold: Minimum sigmoid probability for a pseudo-positive maximum.
        beta: Weight of the mirrored hard-negative term.
        background_weight: Reliability assigned to annotated background.
        gamma_1: Focal exponent for observed positives and regular negatives.
        gamma_2: Focal exponent for mirrored pseudo positives.
        alpha: Positive balancing factor, as in sigmoid focal loss.
        gaussian_weight: Weight of the supervised Gaussian regression term.
        focal_weight: Weight of the focal and mirror-focal terms.
        warmup_epochs: Epochs trained without pseudo positives.
        ramp_epochs: Epochs over which the mirror term reaches full strength.
        max_pseudo_ratio: Multiplier on the propensity-derived pseudo budget.
        local_max_kernel: Odd pooling size used to isolate candidate peaks.
        exclusion_threshold: Candidates cannot overlap an observed target above
            this value.
    """

    outputs_logits = True

    def __init__(
            self,
            drop_rate,
            threshold=None,
            beta=None,
            background_weight=None,
            gamma_1=2.0,
            gamma_2=2.0,
            alpha=0.25,
            gaussian_weight=1.0,
            focal_weight=0.25,
            warmup_epochs=1,
            ramp_epochs=3,
            max_pseudo_ratio=1.0,
            local_max_kernel=5,
            exclusion_threshold=0.10,
            eps=1e-6,
    ):
        super().__init__()
        if not 0 <= drop_rate < 1:
            raise ValueError('drop_rate must be in [0, 1)')
        if local_max_kernel < 1 or local_max_kernel % 2 == 0:
            raise ValueError('local_max_kernel must be a positive odd integer')
        if warmup_epochs < 0 or ramp_epochs < 0:
            raise ValueError('warmup_epochs and ramp_epochs must be non-negative')
        if max_pseudo_ratio < 0:
            raise ValueError('max_pseudo_ratio must be non-negative')
        if not 0 < alpha < 1:
            raise ValueError('alpha must be in (0, 1)')
        if gamma_1 < 0 or gamma_2 < 0:
            raise ValueError('gamma_1 and gamma_2 must be non-negative')
        if gaussian_weight < 0 or focal_weight < 0:
            raise ValueError('loss component weights must be non-negative')
        if not 0 <= exclusion_threshold < 1:
            raise ValueError('exclusion_threshold must be in [0, 1)')

        defaults = partial_annotation_defaults(drop_rate * 100.0)
        self.drop_rate = float(drop_rate)
        self.threshold = defaults['threshold'] if threshold is None else float(threshold)
        self.beta = defaults['beta'] if beta is None else float(beta)
        self.background_weight = (
            defaults['background_weight']
            if background_weight is None else float(background_weight)
        )
        if not 0 < self.threshold < 1:
            raise ValueError('threshold must be in (0, 1)')
        if self.beta < 0 or self.background_weight < 0:
            raise ValueError('beta and background_weight must be non-negative')

        self.gamma_1 = float(gamma_1)
        self.gamma_2 = float(gamma_2)
        self.alpha = float(alpha)
        self.gaussian_weight = float(gaussian_weight)
        self.focal_weight = float(focal_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.max_pseudo_ratio = float(max_pseudo_ratio)
        self.local_max_kernel = int(local_max_kernel)
        self.exclusion_threshold = float(exclusion_threshold)
        self.eps = float(eps)
        self.epoch = 0
        self.last_stats = {}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _mirror_scale(self):
        if not self.training or self.drop_rate == 0 or self.epoch <= self.warmup_epochs:
            return 0.0
        if self.ramp_epochs == 0:
            return 1.0
        return min(1.0, (self.epoch - self.warmup_epochs) / self.ramp_epochs)

    @staticmethod
    def _target_transform(prediction, target, kernel):
        target = F.adaptive_max_pool2d(target, prediction.shape[2:])
        with torch.no_grad():
            transformed = F.conv2d(
                target,
                kernel.to(device=target.device, dtype=target.dtype),
                padding=(kernel.shape[-1] - 1) // 2,
            )
        return transformed.clamp_(0.0, 1.0)

    # Preserve the misspelled helper used by the original trainer's visualizer.
    def _traget_transform(self, prediction, target, kernel):
        return self._target_transform(prediction, target, kernel)

    def _pseudo_peaks(self, probabilities, gaussian_target, sparse_target):
        """Select detached local maxima within the missing-instance budget."""
        selected = torch.zeros_like(probabilities)
        scale = self._mirror_scale()
        if scale == 0.0 or self.max_pseudo_ratio == 0.0:
            return selected, 0

        detached = probabilities.detach()
        pooled = F.max_pool2d(
            detached,
            kernel_size=self.local_max_kernel,
            stride=1,
            padding=self.local_max_kernel // 2,
        )
        candidates = (
            (detached >= self.threshold)
            & (detached >= pooled)
            & (gaussian_target < self.exclusion_threshold)
        )
        hidden_to_observed = self.drop_rate / max(1.0 - self.drop_rate, self.eps)
        selected_count = 0

        # B and C are small in MVDet (batch size is normally one; C is 1 or 2).
        # Per-channel budgets prevent one camera head/foot heatmap monopolizing
        # all pseudo labels.
        for batch_idx in range(probabilities.shape[0]):
            for channel_idx in range(probabilities.shape[1]):
                observed_count = int(
                    (sparse_target[batch_idx, channel_idx] > 0).sum().item()
                )
                expected_count = (
                    observed_count * hidden_to_observed * self.max_pseudo_ratio
                )
                # Round to nearest rather than always rounding up: ceil would
                # systematically invent one person in lightly populated pa20
                # frames where the expected hidden count is below one.
                budget = int(math.floor(expected_count + 0.5))
                candidate_indices = candidates[batch_idx, channel_idx].flatten().nonzero().flatten()
                if budget == 0 or candidate_indices.numel() == 0:
                    continue
                budget = min(budget, candidate_indices.numel())
                scores = detached[batch_idx, channel_idx].flatten()[candidate_indices]
                keep = candidate_indices[torch.topk(scores, k=budget, sorted=False).indices]
                selected[batch_idx, channel_idx].view(-1)[keep] = 1.0
                selected_count += budget

        return selected, selected_count

    def _pseudo_heatmap(self, peaks, kernel, observed_target):
        if not torch.any(peaks):
            return peaks
        with torch.no_grad():
            pseudo = F.conv2d(
                peaks,
                kernel.to(device=peaks.device, dtype=peaks.dtype),
                padding=(kernel.shape[-1] - 1) // 2,
            ).clamp_(0.0, 1.0)
            # Observed supervision always takes precedence over a pseudo label.
            pseudo.mul_(1.0 - observed_target)
        return pseudo

    def forward(self, logits, target, kernel):
        target = target.to(device=logits.device, dtype=logits.dtype)
        sparse_target = F.adaptive_max_pool2d(target, logits.shape[2:])
        gaussian_target = self._target_transform(logits, target, kernel)
        probabilities = torch.sigmoid(logits)
        peaks, num_pseudo_peaks = self._pseudo_peaks(
            probabilities, gaussian_target, sparse_target
        )
        pseudo_target = self._pseudo_heatmap(peaks, kernel, gaussian_target)
        mirror_scale = self._mirror_scale()

        # The Gaussian component retains MVDet's heatmap-shape supervision.
        # Pseudo regions are removed from the negative regression term so the
        # two components cannot push the same location in opposite directions.
        gaussian_positive_weight = gaussian_target
        focal_positive_weight = sparse_target.clamp(0.0, 1.0)
        negative_weight = (1.0 - gaussian_target).pow(4) * (1.0 - pseudo_target)
        effective_background_weight = self.background_weight if self.training else 1.0
        squared_error = (probabilities - gaussian_target).pow(2)
        gaussian_positive = (squared_error * gaussian_positive_weight).sum() / (
            gaussian_positive_weight.sum() + self.eps
        )
        gaussian_negative = (squared_error * negative_weight).sum() / (
            negative_weight.sum() + self.eps
        )
        pseudo_gaussian = (
            (probabilities - pseudo_target).pow(2) * pseudo_target
        ).sum() / (pseudo_target.sum() + self.eps)
        gaussian_loss = (
            gaussian_positive
            + effective_background_weight * gaussian_negative
            + self.beta * mirror_scale * pseudo_gaussian
        )

        # Numerically stable BCE terms. The final term is the mirrored branch:
        # confident unlabeled peaks are optimized as positives rather than as
        # hard negatives, matching the core BRL idea without an anchor assigner.
        positive_bce = F.softplus(-logits)
        negative_bce = F.softplus(logits)
        focal_positive = (
            focal_positive_weight
            * (1.0 - probabilities).pow(self.gamma_1)
            * positive_bce
        ).sum() / (focal_positive_weight.sum() + self.eps)
        focal_negative = (
            negative_weight * probabilities.pow(self.gamma_1) * negative_bce
        ).sum() / (negative_weight.sum() + self.eps)
        mirror_positive = (
            peaks * (1.0 - probabilities).pow(self.gamma_2) * positive_bce
        ).sum() / (peaks.sum() + self.eps)

        focal_loss = (
            self.alpha * focal_positive
            + (1.0 - self.alpha) * effective_background_weight * focal_negative
            + (1.0 - self.alpha) * self.beta * mirror_scale * mirror_positive
        )
        loss = self.gaussian_weight * gaussian_loss + self.focal_weight * focal_loss

        self.last_stats = {
            'mirror_scale': mirror_scale,
            'pseudo_peaks': float(num_pseudo_peaks),
            'pseudo_pixels': float((pseudo_target > 0).sum().item()),
            'gaussian_loss': float(gaussian_loss.detach().item()),
            'pseudo_gaussian_loss': float(pseudo_gaussian.detach().item()),
            'focal_loss': float(focal_loss.detach().item()),
        }
        return loss
