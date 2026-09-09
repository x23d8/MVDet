import math

import torch
import torch.nn.functional as F
from torch import nn

from .heatmap import masked_mean, max_gaussian_target


def project_view_foot_probabilities(
        view_predictions,
        projection_matrices,
        output_shape,
        outputs_logits=True,
        coverage_threshold=0.5,
):
    """Project per-camera foot probabilities and visibility to the BEV grid."""
    try:
        from kornia.geometry.transform import warp_perspective
    except ImportError as exc:
        raise RuntimeError(
            'Projecting camera evidence requires kornia; install requirements.txt'
        ) from exc

    if view_predictions is None or projection_matrices is None:
        raise ValueError('View predictions and projection matrices are required')
    if len(view_predictions) != len(projection_matrices) or not view_predictions:
        raise ValueError('Predictions and matrices must contain the same non-zero number of views')

    projected_scores = []
    projected_visibility = []
    for prediction, projection_matrix in zip(view_predictions, projection_matrices):
        if prediction.ndim != 4 or prediction.shape[1] < 1:
            raise ValueError('Each view prediction must have shape [B, C, H, W]')
        foot_channel = 1 if prediction.shape[1] > 1 else 0
        foot = prediction[:, foot_channel:foot_channel + 1]
        foot_probability = torch.sigmoid(foot) if outputs_logits else foot.clamp(0.0, 1.0)

        batch_size = prediction.shape[0]
        matrix = projection_matrix.to(device=prediction.device, dtype=prediction.dtype)
        matrix = matrix.unsqueeze(0).expand(batch_size, -1, -1)
        projected_scores.append(warp_perspective(foot_probability, matrix, output_shape))

        source_visibility = torch.ones_like(foot_probability)
        visibility = warp_perspective(source_visibility, matrix, output_shape)
        projected_visibility.append(visibility > coverage_threshold)

    return (
        torch.stack(projected_scores, dim=1),
        torch.stack(projected_visibility, dim=1),
    )


def aggregate_multiview_evidence(scores, visibility, topk=2, min_views=2):
    """Return detached top-k consensus, view count and consensus-valid mask."""
    if scores.ndim != 5 or visibility.shape != scores.shape:
        raise ValueError('Projected scores and visibility must have shape [B, N, 1, H, W]')
    if topk < 1 or min_views < 1:
        raise ValueError('topk and min_views must be positive')

    visibility = visibility.bool()
    visible_count = visibility.sum(dim=1)
    effective_topk = min(topk, scores.shape[1])
    masked_scores = scores.masked_fill(~visibility, 0.0)
    top_scores = masked_scores.topk(k=effective_topk, dim=1).values
    consensus = top_scores.mean(dim=1)
    consensus_valid = visible_count >= max(min_views, effective_topk)
    consensus = torch.where(consensus_valid, consensus, torch.zeros_like(consensus))
    return consensus.detach(), visible_count.detach(), consensus_valid.detach()


class BEVBRLLoss(nn.Module):
    """Partial-annotation-aware loss for the final MVDet occupancy logits."""

    requires_multiview_context = True
    outputs_logits = True

    def __init__(
            self,
            positive_threshold=0.10,
            ignore_threshold=0.01,
            negative_threshold=0.15,
            view_negative_threshold=0.15,
            bev_threshold=0.60,
            view_threshold=0.55,
            min_views=2,
            consensus_topk=2,
            local_max_kernel=3,
            focal_alpha=0.25,
            gamma_negative=2.0,
            gamma_mirror=2.0,
            mirror_beta=1.0,
            positive_weight=1.0,
            negative_weight=0.5,
            brl_weight=0.1,
            warmup_epochs=3,
            ramp_epochs=3,
            negative_warmup_factor=0.25,
            coverage_threshold=0.5,
            max_mirror_per_observed=1.5,
            skip_empty_targets=True,
            use_consensus=True,
            mirror_without_consensus=False,
            view_outputs_logits=True,
            eps=1e-6,
    ):
        super().__init__()
        if not 0 <= ignore_threshold < positive_threshold <= 1:
            raise ValueError('Expected 0 <= ignore_threshold < positive_threshold <= 1')
        if not 0 <= negative_threshold < bev_threshold <= 1:
            raise ValueError('Expected 0 <= negative_threshold < bev_threshold <= 1')
        if not 0 <= view_negative_threshold < view_threshold <= 1:
            raise ValueError('Expected 0 <= view_negative_threshold < view_threshold <= 1')
        if min_views < 1 or consensus_topk < 1:
            raise ValueError('min_views and consensus_topk must be positive')
        if local_max_kernel < 1 or local_max_kernel % 2 == 0:
            raise ValueError('local_max_kernel must be a positive odd integer')
        if warmup_epochs < 0 or ramp_epochs < 0:
            raise ValueError('warmup_epochs and ramp_epochs must be non-negative')
        if not 0 <= negative_warmup_factor <= 1:
            raise ValueError('negative_warmup_factor must be in [0, 1]')

        self.positive_threshold = positive_threshold
        self.ignore_threshold = ignore_threshold
        self.negative_threshold = negative_threshold
        self.view_negative_threshold = view_negative_threshold
        self.bev_threshold = bev_threshold
        self.view_threshold = view_threshold
        self.min_views = min_views
        self.consensus_topk = consensus_topk
        self.local_max_kernel = local_max_kernel
        self.focal_alpha = focal_alpha
        self.gamma_negative = gamma_negative
        self.gamma_mirror = gamma_mirror
        self.mirror_beta = mirror_beta
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.brl_weight = brl_weight
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.negative_warmup_factor = negative_warmup_factor
        self.coverage_threshold = coverage_threshold
        self.max_mirror_per_observed = max_mirror_per_observed
        self.skip_empty_targets = skip_empty_targets
        self.use_consensus = use_consensus
        self.mirror_without_consensus = mirror_without_consensus
        self.view_outputs_logits = view_outputs_logits
        self.eps = eps
        self.epoch = 0
        self.last_stats = {}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @property
    def ramp_factor(self):
        if self.epoch <= self.warmup_epochs:
            return 0.0
        if self.ramp_epochs == 0:
            return 1.0
        return min(max((self.epoch - self.warmup_epochs) / self.ramp_epochs, 0.0), 1.0)

    @property
    def current_brl_weight(self):
        return self.brl_weight * self.ramp_factor

    @property
    def current_negative_weight(self):
        scale = self.negative_warmup_factor + (1.0 - self.negative_warmup_factor) * self.ramp_factor
        return self.negative_weight * scale

    def max_gaussian_target(self, prediction, sparse_target, kernel):
        return max_gaussian_target(prediction, sparse_target, kernel)

    def _traget_transform(self, prediction, sparse_target, kernel):
        return self.max_gaussian_target(prediction, sparse_target, kernel)

    def project_view_evidence(self, view_predictions, projection_matrices, output_shape):
        return project_view_foot_probabilities(
            view_predictions,
            projection_matrices,
            output_shape,
            outputs_logits=self.view_outputs_logits,
            coverage_threshold=self.coverage_threshold,
        )

    def _limit_candidates(self, candidate_mask, rank_scores, sparse_target):
        if self.max_mirror_per_observed <= 0:
            return candidate_mask
        pooled_points = F.adaptive_max_pool2d(sparse_target.float(), rank_scores.shape[-2:])
        selected = torch.zeros_like(candidate_mask)
        for batch_idx in range(candidate_mask.shape[0]):
            observed_count = int((pooled_points[batch_idx] > 0).sum().item())
            if observed_count == 0:
                continue
            limit = max(1, int(math.ceil(observed_count * self.max_mirror_per_observed)))
            indices = torch.nonzero(candidate_mask[batch_idx].reshape(-1), as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                continue
            keep_count = min(limit, indices.numel())
            values = rank_scores[batch_idx].reshape(-1)[indices]
            keep = indices[values.topk(keep_count).indices]
            selected[batch_idx].reshape(-1)[keep] = True
        return selected

    def forward(
            self,
            map_logits,
            sparse_target,
            kernel,
            view_predictions=None,
            projection_matrices=None,
            projected_view_scores=None,
            projected_visibility=None,
    ):
        if map_logits.ndim != 4 or map_logits.shape[1] != 1:
            raise ValueError('BEVBRLLoss expects [B, 1, H, W] logits')
        if sparse_target.ndim == 3:
            sparse_target = sparse_target.unsqueeze(1)
        sparse_target = sparse_target.to(map_logits.device)

        probability = torch.sigmoid(map_logits)
        target = self.max_gaussian_target(map_logits, sparse_target, kernel)
        if (projected_view_scores is None) != (projected_visibility is None):
            raise ValueError('Pass both projected_view_scores and projected_visibility')
        if projected_view_scores is None:
            projected_view_scores, projected_visibility = self.project_view_evidence(
                view_predictions,
                projection_matrices,
                map_logits.shape[-2:],
            )

        if self.use_consensus:
            consensus, visible_count, consensus_valid = aggregate_multiview_evidence(
                projected_view_scores,
                projected_visibility,
                topk=self.consensus_topk,
                min_views=self.min_views,
            )
            # Observed annotations remain valid when at least one camera sees
            # the cell. Requiring min_views here would silently drop valid GT;
            # only pseudo-label consensus requires multiple cameras.
            valid_bev = (visible_count >= 1).detach()
        else:
            visible_count = projected_visibility.bool().sum(dim=1).detach()
            valid_bev = (visible_count >= 1).detach()
            consensus = torch.zeros_like(probability).detach()

        detached_probability = probability.detach()
        has_observed = F.adaptive_max_pool2d(sparse_target.float(), 1) > 0
        assignable_sample = has_observed if self.skip_empty_targets else torch.ones_like(has_observed)
        positive_mask = valid_bev & (target >= self.positive_threshold)
        halo_mask = valid_bev & (target > self.ignore_threshold) & ~positive_mask
        unlabeled_mask = valid_bev & (target <= self.ignore_threshold) & assignable_sample

        consensus_low = (
            consensus <= self.view_negative_threshold
            if self.use_consensus else torch.ones_like(unlabeled_mask)
        )
        easy_negative_mask = unlabeled_mask & consensus_low & (
            detached_probability <= self.negative_threshold
        )

        # High predictions are only hard negatives once camera heads have had
        # time to learn.  Before that, low view confidence is not trustworthy.
        hard_negative_mask = torch.zeros_like(unlabeled_mask)
        if self.use_consensus and self.ramp_factor > 0:
            hard_negative_mask = (
                unlabeled_mask
                & (detached_probability >= self.bev_threshold)
                & consensus_low
            )
        reliable_negative_mask = easy_negative_mask | hard_negative_mask

        local_maximum = detached_probability == F.max_pool2d(
            detached_probability,
            kernel_size=self.local_max_kernel,
            stride=1,
            padding=self.local_max_kernel // 2,
        )
        mirror_mask = unlabeled_mask & local_maximum & (
            detached_probability >= self.bev_threshold
        )
        if self.use_consensus:
            mirror_mask &= consensus >= self.view_threshold
            mirror_mask &= visible_count >= self.min_views
        elif not self.mirror_without_consensus:
            mirror_mask = torch.zeros_like(mirror_mask)
        mirror_rank = detached_probability * (consensus if self.use_consensus else 1.0)
        mirror_mask = self._limit_candidates(mirror_mask, mirror_rank, sparse_target)
        uncertain_mask = unlabeled_mask & ~reliable_negative_mask & ~mirror_mask

        positive_element = (probability - target).square()
        negative_element = -(
            (1.0 - self.focal_alpha)
            * probability.pow(self.gamma_negative)
            * torch.log1p(-probability.clamp(max=1.0 - self.eps))
        )
        mirror_element = -(
            self.mirror_beta
            * (1.0 - self.focal_alpha)
            * (1.0 - probability).pow(self.gamma_mirror)
            * torch.log(probability.clamp_min(self.eps))
        )

        loss_positive = masked_mean(positive_element, positive_mask)
        loss_negative = masked_mean(negative_element, reliable_negative_mask)
        loss_mirror = masked_mean(mirror_element, mirror_mask)
        loss = (
            self.positive_weight * loss_positive
            + self.current_negative_weight * loss_negative
            + self.current_brl_weight * loss_mirror
        )

        self.last_stats = {
            'loss_positive': float(loss_positive.detach()),
            'loss_negative': float(loss_negative.detach()),
            'loss_mirror': float(loss_mirror.detach()),
            'negative_weight': float(self.current_negative_weight),
            'brl_weight': float(self.current_brl_weight),
            'positive_cells': float(positive_mask.sum().detach()),
            'halo_cells': float(halo_mask.sum().detach()),
            'reliable_negative_cells': float(reliable_negative_mask.sum().detach()),
            'hard_negative_cells': float(hard_negative_mask.sum().detach()),
            'mirror_cells': float(mirror_mask.sum().detach()),
            'uncertain_cells': float(uncertain_mask.sum().detach()),
            'visible_cells': float(valid_bev.sum().detach()),
            'empty_samples_skipped': float((~has_observed).sum().detach()),
            'max_probability': float(probability.detach().max()),
            'max_consensus': float(consensus.detach().max()),
        }
        return loss
