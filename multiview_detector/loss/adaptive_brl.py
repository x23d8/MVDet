import torch
import torch.nn.functional as F
from torch import nn

from .heatmap import masked_mean, max_gaussian_target


def project_foot_evidence(view_logits, projection_matrices, output_shape, coverage_threshold=0.5):
    """Project foot probabilities and camera visibility to the BEV plane."""
    if not view_logits or len(view_logits) != len(projection_matrices):
        raise ValueError('one projection matrix is required for every non-empty view')
    try:
        from kornia.geometry.transform import warp_perspective
    except ImportError as exc:
        raise RuntimeError('multi-view BRL requires kornia') from exc

    scores, visibility = [], []
    for logits, projection in zip(view_logits, projection_matrices):
        if logits.ndim != 4 or logits.shape[1] < 1:
            raise ValueError('view logits must have shape [B, C, H, W]')
        foot_channel = 1 if logits.shape[1] > 1 else 0
        foot = torch.sigmoid(logits[:, foot_channel:foot_channel + 1])
        matrix = projection.to(device=logits.device, dtype=logits.dtype)
        matrix = matrix.unsqueeze(0).expand(logits.shape[0], -1, -1)
        scores.append(warp_perspective(foot, matrix, output_shape))
        coverage = warp_perspective(torch.ones_like(foot), matrix, output_shape)
        visibility.append(coverage > coverage_threshold)
    return torch.stack(scores, dim=1), torch.stack(visibility, dim=1)


def robust_multiview_consensus(projected_scores, projected_visibility, min_views=2):
    """Top-two agreement score, normalized across different camera counts."""
    if projected_scores.ndim != 5 or projected_visibility.shape != projected_scores.shape:
        raise ValueError('projected tensors must have shape [B, V, C, H, W]')
    if min_views < 1:
        raise ValueError('min_views must be positive')

    visibility = projected_visibility.bool()
    visible_count = visibility.sum(dim=1)
    k = min(2, projected_scores.shape[1])
    top = projected_scores.masked_fill(~visibility, -1.0).topk(k=k, dim=1).values
    if k == 1:
        consensus = top[:, 0]
    else:
        # A high score requires two cameras to agree; disagreement is penalized
        # smoothly rather than through a dataset-specific confidence threshold.
        consensus = top.mean(dim=1) * torch.exp(-(top[:, 0] - top[:, 1]).abs())
    valid = visible_count >= max(min_views, k)
    return consensus.clamp(0.0, 1.0).detach(), valid.detach()


class AdaptiveBRLLoss(nn.Module):
    """Propensity-constrained Background Recalibration Loss for heatmaps.

    Zeros in a partial target are treated as unlabeled. The known annotation
    propensity determines a per-sample pseudo-positive budget; it never fixes a
    dataset-specific score threshold. Evidence may come from another branch or
    from geometrically aligned cameras and is always detached.
    """

    outputs_logits = True

    def __init__(
            self,
            annotation_probability=1.0,
            positive_threshold=0.05,
            ignore_threshold=0.005,
            gamma=2.0,
            background_weight=1.0,
            pseudo_weight=0.25,
            warmup_epochs=1,
            ramp_epochs=3,
            local_max_kernel=3,
            max_pseudo_ratio=1.0,
            eps=1e-6,
    ):
        super().__init__()
        if not 0 < annotation_probability <= 1:
            raise ValueError('annotation_probability must be in (0, 1]')
        if not 0 <= ignore_threshold < positive_threshold <= 1:
            raise ValueError('expected 0 <= ignore_threshold < positive_threshold <= 1')
        if gamma < 0 or background_weight < 0 or pseudo_weight < 0:
            raise ValueError('loss weights and gamma must be non-negative')
        if warmup_epochs < 0 or ramp_epochs < 0:
            raise ValueError('warmup and ramp epochs must be non-negative')
        if local_max_kernel < 1 or local_max_kernel % 2 == 0:
            raise ValueError('local_max_kernel must be a positive odd integer')
        if max_pseudo_ratio <= 0:
            raise ValueError('max_pseudo_ratio must be positive')

        self.annotation_probability = float(annotation_probability)
        self.positive_threshold = positive_threshold
        self.ignore_threshold = ignore_threshold
        self.gamma = gamma
        self.background_weight = background_weight
        self.pseudo_weight = pseudo_weight
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.local_max_kernel = local_max_kernel
        self.max_pseudo_ratio = max_pseudo_ratio
        self.eps = eps
        self.epoch = 0
        self.last_stats = {}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _traget_transform(self, prediction, sparse_target, kernel):
        # Keep the historical misspelled hook used by MVDet visualization.
        return max_gaussian_target(prediction, sparse_target, kernel)

    @property
    def ramp_factor(self):
        if self.epoch <= self.warmup_epochs:
            return 0.0
        if self.ramp_epochs == 0:
            return 1.0
        return min((self.epoch - self.warmup_epochs) / self.ramp_epochs, 1.0)

    def _pseudo_targets(self, prediction, sparse_target, kernel, evidence, validity, observed_heatmap):
        pooled = F.adaptive_max_pool2d(sparse_target.float(), prediction.shape[-2:])
        if pooled.shape[1] == 1 and prediction.shape[1] != 1:
            pooled = pooled.expand(-1, prediction.shape[1], -1, -1)
        candidate_mask = (observed_heatmap <= self.ignore_threshold) & validity
        local_max = F.max_pool2d(
            evidence,
            self.local_max_kernel,
            stride=1,
            padding=self.local_max_kernel // 2,
        )
        candidate_mask &= evidence >= local_max - self.eps

        points = torch.zeros_like(prediction)
        confidence_points = torch.zeros_like(prediction)
        selected_total = 0
        expected_missing_total = 0.0
        for batch_idx in range(prediction.shape[0]):
            for channel_idx in range(prediction.shape[1]):
                observed_count = int((pooled[batch_idx, channel_idx] > 0).sum().item())
                expected_missing = observed_count * (
                    1.0 - self.annotation_probability
                ) / self.annotation_probability
                expected_missing_total += expected_missing
                # Match the simulator's per-frame round-to-nearest policy. This
                # avoids inventing one pseudo point for every low-count frame
                # when the expected missing mass is substantially below one.
                budget = int(round(expected_missing * self.max_pseudo_ratio))
                if budget == 0 or observed_count == 0:
                    continue
                flat_indices = torch.nonzero(
                    candidate_mask[batch_idx, channel_idx].reshape(-1), as_tuple=False
                ).squeeze(1)
                if flat_indices.numel() == 0:
                    continue
                scores = evidence[batch_idx, channel_idx].reshape(-1)[flat_indices]
                observed_indices = torch.nonzero(
                    pooled[batch_idx, channel_idx].reshape(-1) > 0, as_tuple=False
                ).squeeze(1)
                reference = torch.quantile(
                    evidence[batch_idx, channel_idx].reshape(-1)[observed_indices], 0.25
                )
                calibrated = scores > reference + self.eps
                flat_indices, scores = flat_indices[calibrated], scores[calibrated]
                if flat_indices.numel() == 0:
                    continue
                keep_count = min(budget, flat_indices.numel())
                kept = flat_indices[scores.topk(keep_count).indices]
                points[batch_idx, channel_idx].reshape(-1)[kept] = 1.0
                # Squared probability is a conservative, smoothly calibrated
                # reliability weight (0.5 evidence contributes only 0.25).
                confidence_points[batch_idx, channel_idx].reshape(-1)[kept] = (
                    evidence[batch_idx, channel_idx].reshape(-1)[kept].square()
                )
                selected_total += keep_count

        pseudo_target = max_gaussian_target(prediction, points, kernel)
        pseudo_confidence = max_gaussian_target(prediction, confidence_points, kernel)
        return pseudo_target, pseudo_confidence, selected_total, expected_missing_total

    def forward(self, prediction, sparse_target, kernel, evidence=None, validity=None):
        if prediction.ndim != 4:
            raise ValueError('prediction must have shape [B, C, H, W]')
        observed = max_gaussian_target(prediction, sparse_target, kernel)
        probability = torch.sigmoid(prediction)
        if evidence is None:
            evidence = probability.detach()
        else:
            evidence = evidence.to(device=prediction.device, dtype=prediction.dtype).detach()
            if evidence.shape != prediction.shape:
                raise ValueError('evidence shape must match prediction shape')
        if validity is None:
            validity = torch.ones_like(prediction, dtype=torch.bool)
        else:
            validity = validity.to(device=prediction.device, dtype=torch.bool)
            if validity.shape != prediction.shape:
                raise ValueError('validity shape must match prediction shape')

        pseudo_target, pseudo_confidence, selected, expected_missing = self._pseudo_targets(
            prediction, sparse_target.to(prediction.device), kernel, evidence, validity, observed
        )
        if self.ramp_factor == 0:
            pseudo_target = torch.zeros_like(pseudo_target)
            pseudo_confidence = torch.zeros_like(pseudo_confidence)
        positive_mask = (observed > self.positive_threshold) & validity
        pseudo_mask = (pseudo_target > self.positive_threshold) & ~positive_mask & validity
        ignore_mask = (observed > self.ignore_threshold) | (pseudo_target > self.ignore_threshold)
        negative_mask = ~ignore_mask & validity

        observed_bce = F.binary_cross_entropy_with_logits(prediction, observed, reduction='none')
        observed_modulation = (observed - probability).abs().pow(self.gamma)
        positive_loss = masked_mean(observed_bce * observed_modulation, positive_mask)

        pseudo_bce = F.binary_cross_entropy_with_logits(prediction, pseudo_target, reduction='none')
        pseudo_modulation = (pseudo_target - probability).abs().pow(self.gamma)
        pseudo_loss = masked_mean(
            pseudo_confidence * pseudo_bce * pseudo_modulation,
            pseudo_mask,
        )

        negative_focal = probability.pow(self.gamma) * F.softplus(prediction)
        # BRL/soft-sampling term: independent evidence reduces the false-negative
        # gradient continuously. The count budget still prevents unlimited
        # predictions from becoming pseudo positives.
        negative_reliability = (1.0 - self.ramp_factor * evidence.square()).clamp_min(0.0)
        negative_loss = masked_mean(negative_focal * negative_reliability, negative_mask)

        total = (
            positive_loss
            + self.background_weight * negative_loss
            + self.pseudo_weight * self.ramp_factor * pseudo_loss
        )
        self.last_stats = {
            'positive_loss': float(positive_loss.detach()),
            'negative_loss': float(negative_loss.detach()),
            'pseudo_loss': float(pseudo_loss.detach()),
            'positive_cells': float(positive_mask.sum().detach()),
            'negative_cells': float(negative_mask.sum().detach()),
            'pseudo_cells': float(pseudo_mask.sum().detach()),
            'pseudo_points': float(selected if self.ramp_factor > 0 else 0),
            'expected_missing_points': float(expected_missing),
            'ramp_factor': float(self.ramp_factor),
        }
        return total


class MultiViewAdaptiveBRLLoss(AdaptiveBRLLoss):
    """Adaptive BRL using cross-camera foot evidence for BEV occupancy."""

    requires_multiview_context = True

    def __init__(self, *args, min_views=2, coverage_threshold=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if min_views < 1:
            raise ValueError('min_views must be positive')
        self.min_views = min_views
        self.coverage_threshold = coverage_threshold

    def forward(self, prediction, sparse_target, kernel, view_logits=None, projection_matrices=None):
        if view_logits is None or projection_matrices is None:
            return super().forward(prediction, sparse_target, kernel)
        scores, visibility = project_foot_evidence(
            view_logits,
            projection_matrices,
            prediction.shape[-2:],
            self.coverage_threshold,
        )
        view_evidence, valid = robust_multiview_consensus(scores, visibility, self.min_views)
        # Two independently trained heads must agree before a missing point is
        # promoted. Geometric mean is conservative and scale symmetric.
        map_evidence = torch.sqrt(
            torch.sigmoid(prediction).detach().clamp_min(self.eps)
            * view_evidence.clamp_min(self.eps)
        )
        # Do not let a single camera (or the BEV head alone) confirm itself.
        # The self-evidence fallback is reserved for variants that have no
        # learned camera head and therefore instantiate AdaptiveBRLLoss instead.
        evidence = torch.where(valid, map_evidence, torch.zeros_like(map_evidence))
        validity = visibility.any(dim=1)
        return super().forward(prediction, sparse_target, kernel, evidence, validity)
