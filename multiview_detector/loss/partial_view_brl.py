import math

import torch
import torch.nn.functional as F
from torch import nn

from .bev_brl import aggregate_multiview_evidence
from .heatmap import masked_mean, max_gaussian_target


class PartialViewBRLLoss(nn.Module):
    """Partial-aware auxiliary loss for MVDet head/foot camera heatmaps.

    Observed head and foot points are supervised with max-Gaussian targets.
    Unlabelled head pixels are ignored by default because a BEV ground point
    cannot determine head height.  Foot negatives and pseudo positives require
    detached fused-BEV evidence plus leave-one-view-out camera agreement.
    """

    outputs_logits = True
    requires_bev_context = True

    def __init__(
            self,
            positive_threshold=0.10,
            ignore_threshold=0.01,
            negative_threshold=0.15,
            hard_negative_threshold=0.40,
            pseudo_threshold=0.60,
            support_negative_threshold=0.30,
            support_positive_threshold=0.65,
            min_views=2,
            consensus_topk=2,
            local_max_kernel=3,
            focal_alpha=0.25,
            gamma_negative=2.0,
            gamma_pseudo=2.0,
            positive_weight=1.0,
            negative_weight=0.25,
            hard_negative_weight=0.5,
            pseudo_weight=0.10,
            head_weight=0.05,
            foot_weight=1.0,
            head_negative_weight=0.0,
            warmup_epochs=1,
            ramp_epochs=2,
            coverage_threshold=0.5,
            max_pseudo_per_observed=1.0,
            skip_empty_targets=True,
            eps=1e-6,
    ):
        super().__init__()
        if not 0 <= ignore_threshold < positive_threshold <= 1:
            raise ValueError('Expected 0 <= ignore_threshold < positive_threshold <= 1')
        if not 0 <= negative_threshold < hard_negative_threshold <= 1:
            raise ValueError('Expected 0 <= negative_threshold < hard_negative_threshold <= 1')
        if not 0 <= support_negative_threshold < support_positive_threshold <= 1:
            raise ValueError('Expected support negative threshold < support positive threshold')
        if not 0 < pseudo_threshold <= 1:
            raise ValueError('pseudo_threshold must be in (0, 1]')
        if min_views < 1 or consensus_topk < 1:
            raise ValueError('min_views and consensus_topk must be positive')
        if local_max_kernel < 1 or local_max_kernel % 2 == 0:
            raise ValueError('local_max_kernel must be a positive odd integer')
        if warmup_epochs < 0 or ramp_epochs < 0:
            raise ValueError('warmup_epochs and ramp_epochs must be non-negative')

        self.positive_threshold = positive_threshold
        self.ignore_threshold = ignore_threshold
        self.negative_threshold = negative_threshold
        self.hard_negative_threshold = hard_negative_threshold
        self.pseudo_threshold = pseudo_threshold
        self.support_negative_threshold = support_negative_threshold
        self.support_positive_threshold = support_positive_threshold
        self.min_views = min_views
        self.consensus_topk = consensus_topk
        self.local_max_kernel = local_max_kernel
        self.focal_alpha = focal_alpha
        self.gamma_negative = gamma_negative
        self.gamma_pseudo = gamma_pseudo
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.hard_negative_weight = hard_negative_weight
        self.pseudo_weight = pseudo_weight
        self.head_weight = head_weight
        self.foot_weight = foot_weight
        self.head_negative_weight = head_negative_weight
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.coverage_threshold = coverage_threshold
        self.max_pseudo_per_observed = max_pseudo_per_observed
        self.skip_empty_targets = skip_empty_targets
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

    def _build_foot_support(
            self,
            view_logits,
            map_logits,
            projection_matrices,
            projected_scores,
            projected_visibility,
    ):
        """Back-project detached BEV/other-view agreement to each camera."""
        try:
            from kornia.geometry.transform import warp_perspective
        except ImportError as exc:
            raise RuntimeError(
                'Partial view BRL projection requires kornia; install requirements.txt'
            ) from exc

        num_views = len(view_logits)
        if num_views < 2:
            raise ValueError('Foot support requires at least two camera views')
        if len(projection_matrices) != num_views:
            raise ValueError('Projection matrix count does not match view count')
        if projected_scores.shape[1] != num_views:
            raise ValueError('Projected score count does not match view count')

        map_probability = torch.sigmoid(map_logits).detach()
        scores = projected_scores.detach()
        visibility = projected_visibility.bool().detach()
        support_images = []
        support_validity = []

        # If global consensus uses K views, leave-one-out support needs K-1
        # other views.  This prevents a camera from confirming itself.
        other_topk = max(1, self.consensus_topk - 1)
        other_min_views = max(1, self.min_views - 1)
        for view_idx, prediction in enumerate(view_logits):
            other_indices = [idx for idx in range(num_views) if idx != view_idx]
            other_scores = scores[:, other_indices]
            other_visibility = visibility[:, other_indices]
            consensus, _, consensus_valid = aggregate_multiview_evidence(
                other_scores,
                other_visibility,
                topk=other_topk,
                min_views=other_min_views,
            )

            current_visible = visibility[:, view_idx]
            valid_bev = consensus_valid & current_visible
            support_bev = torch.minimum(map_probability, consensus)
            support_bev = support_bev.masked_fill(~valid_bev, 0.0)

            batch_size = prediction.shape[0]
            matrix = projection_matrices[view_idx].to(
                device=prediction.device,
                dtype=prediction.dtype,
            )
            inverse_matrix = torch.linalg.inv(matrix).unsqueeze(0).expand(batch_size, -1, -1)
            output_shape = prediction.shape[-2:]
            support_image = warp_perspective(support_bev, inverse_matrix, output_shape)
            valid_image = warp_perspective(valid_bev.float(), inverse_matrix, output_shape)
            support_images.append(support_image.detach())
            support_validity.append((valid_image > self.coverage_threshold).detach())

        return support_images, support_validity

    def _limit_pseudo(self, mask, rank_score, sparse_foot_target):
        if self.max_pseudo_per_observed <= 0:
            return mask
        pooled = F.adaptive_max_pool2d(sparse_foot_target.float(), rank_score.shape[-2:])
        selected = torch.zeros_like(mask)
        for batch_idx in range(mask.shape[0]):
            observed = int((pooled[batch_idx] > 0).sum().item())
            if observed == 0:
                continue
            limit = max(1, int(math.ceil(observed * self.max_pseudo_per_observed)))
            indices = torch.nonzero(mask[batch_idx].reshape(-1), as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                continue
            keep_count = min(limit, indices.numel())
            values = rank_score[batch_idx].reshape(-1)[indices]
            keep = indices[values.topk(keep_count).indices]
            selected[batch_idx].reshape(-1)[keep] = True
        return selected

    def forward(
            self,
            view_logits,
            sparse_targets,
            kernel,
            map_logits=None,
            projection_matrices=None,
            projected_scores=None,
            projected_visibility=None,
            foot_support_images=None,
            foot_support_validity=None,
    ):
        if len(view_logits) != len(sparse_targets) or not view_logits:
            raise ValueError('View predictions and targets must have the same non-zero length')
        for prediction in view_logits:
            if prediction.ndim != 4 or prediction.shape[1] != 2:
                raise ValueError('PartialViewBRLLoss expects two-channel [head, foot] logits')

        if (foot_support_images is None) != (foot_support_validity is None):
            raise ValueError('Pass both foot_support_images and foot_support_validity')
        if foot_support_images is None:
            required = (map_logits, projection_matrices, projected_scores, projected_visibility)
            if any(value is None for value in required):
                raise ValueError('BEV logits, projection matrices and projected evidence are required')
            foot_support_images, foot_support_validity = self._build_foot_support(
                view_logits,
                map_logits,
                projection_matrices,
                projected_scores,
                projected_visibility,
            )
        if (len(foot_support_images) != len(view_logits)
                or len(foot_support_validity) != len(view_logits)):
            raise ValueError('Foot support count does not match view count')

        total_loss = view_logits[0].new_zeros(())
        totals = {
            'loss_head_positive': 0.0,
            'loss_head_negative': 0.0,
            'loss_foot_positive': 0.0,
            'loss_foot_negative': 0.0,
            'loss_foot_hard_negative': 0.0,
            'loss_foot_pseudo': 0.0,
            'head_positive_cells': 0.0,
            'head_negative_cells': 0.0,
            'head_uncertain_cells': 0.0,
            'foot_positive_cells': 0.0,
            'foot_conservative_negative_cells': 0.0,
            'foot_evidence_negative_cells': 0.0,
            'foot_negative_cells': 0.0,
            'foot_hard_negative_cells': 0.0,
            'foot_pseudo_cells': 0.0,
            'foot_uncertain_cells': 0.0,
            'empty_views_skipped': 0.0,
        }

        for prediction, sparse_target, support, support_valid in zip(
                view_logits,
                sparse_targets,
                foot_support_images,
                foot_support_validity,
        ):
            sparse_target = sparse_target.to(prediction.device)
            if support.shape != prediction[:, 0:1].shape or support_valid.shape != support.shape:
                raise ValueError('Each foot support mask must match [B, 1, H, W] view shape')
            target = max_gaussian_target(prediction, sparse_target, kernel)
            probability = torch.sigmoid(prediction)
            detached_probability = probability.detach()

            positive = target >= self.positive_threshold
            unlabeled = target <= self.ignore_threshold

            head_probability = probability[:, 0:1]
            foot_probability = probability[:, 1:2]
            detached_head = detached_probability[:, 0:1]
            detached_foot = detached_probability[:, 1:2]
            head_positive = positive[:, 0:1]
            foot_positive = positive[:, 1:2]
            head_unlabeled = unlabeled[:, 0:1]
            foot_unlabeled = unlabeled[:, 1:2]
            has_observed_head = F.adaptive_max_pool2d(
                sparse_target[:, 0:1].float(), 1
            ) > 0
            has_observed_foot = F.adaptive_max_pool2d(
                sparse_target[:, 1:2].float(), 1
            ) > 0
            assignable_head = (
                has_observed_head if self.skip_empty_targets
                else torch.ones_like(has_observed_head)
            )
            assignable_foot = (
                has_observed_foot if self.skip_empty_targets
                else torch.ones_like(has_observed_foot)
            )
            head_unlabeled = head_unlabeled & assignable_head
            foot_unlabeled = foot_unlabeled & assignable_foot

            # A ground point cannot locate the head without a height model.
            # Default head_negative_weight=0 therefore makes all unlabelled
            # head pixels unknown rather than false background labels.
            head_negative = torch.zeros_like(head_unlabeled)
            if self.head_negative_weight > 0 and self.ramp_factor > 0:
                head_negative = head_unlabeled & (detached_head <= self.negative_threshold)
            head_uncertain = head_unlabeled & ~head_negative

            support = support.to(device=prediction.device, dtype=prediction.dtype).detach()
            support_valid = support_valid.to(device=prediction.device).bool().detach()
            low_support = support <= self.support_negative_threshold
            high_support = support >= self.support_positive_threshold

            foot_conservative_negative = (
                foot_unlabeled
                & support_valid
                & low_support
                & (detached_foot <= self.negative_threshold)
            )
            foot_evidence_negative = foot_unlabeled & support_valid & low_support
            foot_negative = foot_conservative_negative
            foot_hard_negative = torch.zeros_like(foot_conservative_negative)
            foot_pseudo = torch.zeros_like(foot_conservative_negative)
            if self.ramp_factor > 0:
                # External BEV/other-view support decides negative
                # eligibility. The current foot probability only identifies
                # the hard subset; it no longer lets medium-confidence false
                # positives escape the negative loss.
                foot_negative = foot_evidence_negative
                foot_hard_negative = foot_evidence_negative & (
                    detached_foot >= self.hard_negative_threshold
                )
                local_maximum = detached_foot == F.max_pool2d(
                    detached_foot,
                    kernel_size=self.local_max_kernel,
                    stride=1,
                    padding=self.local_max_kernel // 2,
                )
                foot_pseudo = (
                    foot_unlabeled
                    & support_valid
                    & high_support
                    & local_maximum
                    & (detached_foot >= self.pseudo_threshold)
                )
                foot_pseudo = self._limit_pseudo(
                    foot_pseudo,
                    detached_foot * support,
                    sparse_target[:, 1:2],
                )
            foot_uncertain = foot_unlabeled & ~foot_negative & ~foot_pseudo

            # Use the same non-saturating positive objective as the BEV head.
            # This is especially important for the foot channel because its
            # projected probabilities provide the independent BEV evidence.
            positive_element = F.binary_cross_entropy_with_logits(
                prediction,
                target,
                reduction='none',
            )
            head_negative_element = -(
                (1.0 - self.focal_alpha)
                * head_probability.pow(self.gamma_negative)
                * torch.log1p(-head_probability.clamp(max=1.0 - self.eps))
            )
            foot_negative_element = -(
                (1.0 - self.focal_alpha)
                * foot_probability.pow(self.gamma_negative)
                * torch.log1p(-foot_probability.clamp(max=1.0 - self.eps))
            )
            foot_pseudo_element = -(
                (1.0 - self.focal_alpha)
                * (1.0 - foot_probability).pow(self.gamma_pseudo)
                * torch.log(foot_probability.clamp_min(self.eps))
            )

            loss_head_positive = masked_mean(positive_element[:, 0:1], head_positive)
            loss_foot_positive = masked_mean(positive_element[:, 1:2], foot_positive)
            loss_head_negative = masked_mean(head_negative_element, head_negative)
            loss_foot_negative = masked_mean(foot_negative_element, foot_negative)
            loss_foot_hard_negative = masked_mean(
                foot_negative_element,
                foot_hard_negative,
            )
            loss_foot_pseudo = masked_mean(foot_pseudo_element, foot_pseudo)

            head_loss = (
                self.positive_weight * loss_head_positive
                + self.head_negative_weight * self.ramp_factor * loss_head_negative
            )
            foot_loss = (
                self.positive_weight * loss_foot_positive
                + self.negative_weight * self.ramp_factor * loss_foot_negative
                + self.hard_negative_weight * self.ramp_factor * loss_foot_hard_negative
                + self.pseudo_weight * self.ramp_factor * loss_foot_pseudo
            )
            total_loss = total_loss + self.head_weight * head_loss + self.foot_weight * foot_loss

            tensor_metrics = {
                'loss_head_positive': loss_head_positive,
                'loss_head_negative': loss_head_negative,
                'loss_foot_positive': loss_foot_positive,
                'loss_foot_negative': loss_foot_negative,
                'loss_foot_hard_negative': loss_foot_hard_negative,
                'loss_foot_pseudo': loss_foot_pseudo,
                'head_positive_cells': head_positive.sum(),
                'head_negative_cells': head_negative.sum(),
                'head_uncertain_cells': head_uncertain.sum(),
                'foot_positive_cells': foot_positive.sum(),
                'foot_conservative_negative_cells': foot_conservative_negative.sum(),
                'foot_evidence_negative_cells': foot_evidence_negative.sum(),
                'foot_negative_cells': foot_negative.sum(),
                'foot_hard_negative_cells': foot_hard_negative.sum(),
                'foot_pseudo_cells': foot_pseudo.sum(),
                'foot_uncertain_cells': foot_uncertain.sum(),
                'empty_views_skipped': (~has_observed_foot).sum(),
            }
            for name, value in tensor_metrics.items():
                totals[name] += float(value.detach())

        num_views = len(view_logits)
        total_loss = total_loss / num_views
        for name in totals:
            totals[name] /= num_views
        totals['ramp_factor'] = float(self.ramp_factor)
        totals['max_foot_support'] = max(float(value.detach().max()) for value in foot_support_images)
        self.last_stats = totals
        return total_loss
