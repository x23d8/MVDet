import math

import torch
import torch.nn.functional as F
from torch import nn


class BEVBRLLoss(nn.Module):
    """Background recalibration loss for MVDet ground-plane heatmaps.

    The loss keeps the original BRL idea (easy unlabeled locations are
    negatives, confident unlabeled locations are mirrored) but adapts sample
    assignment to point-based BEV detection:

    * observed points are expanded with a max-Gaussian target;
    * camera coverage excludes unseen ground-plane cells;
    * projected per-view foot heatmaps provide a detached top-k consensus;
    * only local maxima with strong BEV and view evidence are mirrored;
    * samples between the reliable-negative and mirror thresholds are ignored.

    ``map_logits`` must be logits. The current MVDet per-view heads remain
    Gaussian-regression heads by default, so their values are clamped to
    (0, 1] before consensus instead of being passed through a sigmoid.
    """

    requires_multiview_context = True
    outputs_logits = True

    def __init__(
        self,
        positive_threshold=0.1,
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
        coverage_threshold=0.5,
        max_mirror_per_observed=1.5,
        use_consensus=True,
        view_outputs_logits=False,
        eps=1e-6,
    ):
        super().__init__()
        if not 0 <= ignore_threshold < positive_threshold <= 1:
            raise ValueError("Expected 0 <= ignore_threshold < positive_threshold <= 1")
        if not 0 <= negative_threshold < bev_threshold <= 1:
            raise ValueError("Expected 0 <= negative_threshold < bev_threshold <= 1")
        if not 0 <= view_negative_threshold < view_threshold <= 1:
            raise ValueError("Expected 0 <= view_negative_threshold < view_threshold <= 1")
        if min_views < 1:
            raise ValueError("min_views must be positive")
        if consensus_topk < 1:
            raise ValueError("consensus_topk must be positive")
        if local_max_kernel < 1 or local_max_kernel % 2 == 0:
            raise ValueError("local_max_kernel must be a positive odd integer")
        if warmup_epochs < 0 or ramp_epochs < 0:
            raise ValueError("warmup_epochs and ramp_epochs must be non-negative")

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
        self.coverage_threshold = coverage_threshold
        self.max_mirror_per_observed = max_mirror_per_observed
        self.use_consensus = use_consensus
        self.view_outputs_logits = view_outputs_logits
        self.eps = eps
        self.epoch = 0
        self.last_stats = {}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @property
    def current_brl_weight(self):
        if self.epoch <= self.warmup_epochs:
            return 0.0
        if self.ramp_epochs == 0:
            return self.brl_weight
        progress = min(max((self.epoch - self.warmup_epochs) / self.ramp_epochs, 0.0), 1.0)
        return self.brl_weight * progress

    @staticmethod
    def _diagonal_kernel(kernel, channel):
        if kernel.ndim == 2:
            return kernel
        if kernel.ndim != 4:
            raise ValueError("Gaussian kernel must have shape [H, W] or [C, C, H, W]")
        kernel_channel = min(channel, kernel.shape[0] - 1)
        input_channel = min(channel, kernel.shape[1] - 1)
        return kernel[kernel_channel, input_channel]

    def max_gaussian_target(self, prediction, sparse_target, kernel):
        """Expand sparse point targets using max instead of Gaussian sums."""
        if sparse_target.ndim == 3:
            sparse_target = sparse_target.unsqueeze(1)
        if prediction.ndim != 4 or sparse_target.ndim != 4:
            raise ValueError("prediction and sparse_target must have shape [B, C, H, W]")

        pooled = F.adaptive_max_pool2d(sparse_target.float(), prediction.shape[-2:])
        if pooled.shape[1] != prediction.shape[1]:
            if pooled.shape[1] == 1:
                pooled = pooled.expand(-1, prediction.shape[1], -1, -1)
            else:
                raise ValueError("Target channels do not match prediction channels")

        target = torch.zeros_like(prediction)
        kernel = kernel.to(device=prediction.device, dtype=prediction.dtype)
        height, width = prediction.shape[-2:]

        with torch.no_grad():
            for batch_idx in range(prediction.shape[0]):
                for channel_idx in range(prediction.shape[1]):
                    gaussian = self._diagonal_kernel(kernel, channel_idx)
                    kernel_h, kernel_w = gaussian.shape[-2:]
                    center_y, center_x = kernel_h // 2, kernel_w // 2
                    point_indices = torch.nonzero(
                        pooled[batch_idx, channel_idx] > 0,
                        as_tuple=False,
                    )
                    for point_y, point_x in point_indices.tolist():
                        out_y0 = max(point_y - center_y, 0)
                        out_x0 = max(point_x - center_x, 0)
                        out_y1 = min(point_y - center_y + kernel_h, height)
                        out_x1 = min(point_x - center_x + kernel_w, width)

                        kernel_y0 = out_y0 - (point_y - center_y)
                        kernel_x0 = out_x0 - (point_x - center_x)
                        kernel_y1 = kernel_y0 + (out_y1 - out_y0)
                        kernel_x1 = kernel_x0 + (out_x1 - out_x0)

                        current = target[batch_idx, channel_idx, out_y0:out_y1, out_x0:out_x1]
                        patch = gaussian[kernel_y0:kernel_y1, kernel_x0:kernel_x1]
                        target[batch_idx, channel_idx, out_y0:out_y1, out_x0:out_x1] = torch.maximum(
                            current,
                            patch,
                        )
        return target.clamp_(0.0, 1.0)

    # Keep the historical misspelled method name so trainer visualization can
    # use either GaussianMSE or BEVBRLLoss without special casing.
    def _traget_transform(self, prediction, sparse_target, kernel):
        return self.max_gaussian_target(prediction, sparse_target, kernel)

    def _project_view_evidence(self, view_predictions, projection_matrices, output_shape):
        try:
            from kornia.geometry.transform import warp_perspective
        except ImportError as exc:
            raise RuntimeError(
                "BEV-BRL camera projection requires kornia; install the project requirements first"
            ) from exc
        if view_predictions is None or projection_matrices is None:
            raise ValueError("BEV-BRL coverage requires view predictions and projection matrices")
        if len(view_predictions) != len(projection_matrices):
            raise ValueError("The number of view predictions and projection matrices must match")
        if not view_predictions:
            raise ValueError("At least one camera view is required")

        projected_scores = []
        projected_visibility = []
        for view_prediction, projection_matrix in zip(view_predictions, projection_matrices):
            if view_prediction.shape[1] < 1:
                raise ValueError("A view prediction must contain at least one heatmap channel")
            foot_channel = 1 if view_prediction.shape[1] > 1 else 0
            foot_score = view_prediction[:, foot_channel:foot_channel + 1]
            if self.view_outputs_logits:
                foot_score = torch.sigmoid(foot_score)
            else:
                foot_score = foot_score.clamp(0.0, 1.0)

            batch_size = foot_score.shape[0]
            matrix = projection_matrix.to(device=foot_score.device, dtype=foot_score.dtype)
            matrix = matrix.unsqueeze(0).expand(batch_size, -1, -1)
            projected_scores.append(warp_perspective(foot_score, matrix, output_shape))

            source_visibility = torch.ones_like(foot_score)
            visibility = warp_perspective(source_visibility, matrix, output_shape)
            projected_visibility.append(visibility > self.coverage_threshold)

        scores = torch.stack(projected_scores, dim=1)
        visibility = torch.stack(projected_visibility, dim=1)
        return self._aggregate_projected_evidence(scores, visibility)

    def _aggregate_projected_evidence(self, scores, visibility):
        """Aggregate already projected tensors of shape [B, N, 1, H, W]."""
        if scores.ndim != 5 or visibility.shape != scores.shape:
            raise ValueError("Projected scores and visibility must have shape [B, N, 1, H, W]")
        visibility = visibility.bool()
        visible_count = visibility.sum(dim=1)
        valid_bev = visible_count >= 1

        if self.use_consensus:
            topk = min(self.consensus_topk, scores.shape[1])
            masked_scores = scores.masked_fill(~visibility, 0.0)
            top_scores = masked_scores.topk(k=topk, dim=1).values
            consensus = top_scores.mean(dim=1)
            consensus_valid = visible_count >= max(self.min_views, topk)
            consensus = torch.where(consensus_valid, consensus, torch.zeros_like(consensus))
        else:
            consensus = torch.zeros_like(scores[:, 0])

        return consensus.detach(), visible_count.detach(), valid_bev.detach()

    def _limit_mirror_candidates(self, mirror_mask, scores, sparse_target):
        if self.max_mirror_per_observed <= 0:
            return mirror_mask

        pooled_points = F.adaptive_max_pool2d(sparse_target.float(), scores.shape[-2:])
        selected = torch.zeros_like(mirror_mask)
        for batch_idx in range(mirror_mask.shape[0]):
            observed_count = int((pooled_points[batch_idx] > 0).sum().item())
            if observed_count == 0:
                continue
            limit = max(1, int(math.ceil(observed_count * self.max_mirror_per_observed)))
            candidate_indices = torch.nonzero(mirror_mask[batch_idx].reshape(-1), as_tuple=False).squeeze(1)
            if candidate_indices.numel() == 0:
                continue
            flat_scores = scores[batch_idx].reshape(-1)[candidate_indices]
            keep_count = min(limit, candidate_indices.numel())
            keep = candidate_indices[flat_scores.topk(keep_count).indices]
            selected[batch_idx].reshape(-1)[keep] = True
        return selected

    def _masked_mean(self, values, mask):
        mask = mask.to(values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

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
        if map_logits.shape[1] != 1:
            raise ValueError("BEVBRLLoss expects a one-channel ground-plane output")
        if sparse_target.ndim == 3:
            sparse_target = sparse_target.unsqueeze(1)

        probability = torch.sigmoid(map_logits)
        gaussian_target = self.max_gaussian_target(map_logits, sparse_target, kernel)
        if projected_view_scores is not None or projected_visibility is not None:
            if projected_view_scores is None or projected_visibility is None:
                raise ValueError("Pass both projected_view_scores and projected_visibility")
            consensus, visible_count, valid_bev = self._aggregate_projected_evidence(
                projected_view_scores,
                projected_visibility,
            )
        else:
            consensus, visible_count, valid_bev = self._project_view_evidence(
                view_predictions,
                projection_matrices,
                map_logits.shape[-2:],
            )

        detached_probability = probability.detach()
        positive_mask = valid_bev & (gaussian_target >= self.positive_threshold)
        ignore_mask = valid_bev & (gaussian_target > self.ignore_threshold) & ~positive_mask
        unlabeled_mask = valid_bev & (gaussian_target <= self.ignore_threshold)

        easy_negative_mask = (
            unlabeled_mask
            & (detached_probability <= self.negative_threshold)
            & (
                (consensus <= self.view_negative_threshold)
                if self.use_consensus
                else torch.ones_like(unlabeled_mask)
            )
        )

        # A high fused-BEV response without even low-threshold support from
        # the projected view heads is treated as a hard negative. This keeps
        # one-view projection artefacts from being silently ignored forever.
        hard_negative_mask = torch.zeros_like(unlabeled_mask)
        if self.use_consensus:
            hard_negative_mask = (
                unlabeled_mask
                & (detached_probability >= self.bev_threshold)
                & (consensus <= self.view_negative_threshold)
            )
        reliable_negative_mask = easy_negative_mask | hard_negative_mask

        local_maximum = detached_probability == F.max_pool2d(
            detached_probability,
            kernel_size=self.local_max_kernel,
            stride=1,
            padding=self.local_max_kernel // 2,
        )
        mirror_mask = unlabeled_mask & local_maximum & (detached_probability >= self.bev_threshold)
        if self.use_consensus:
            mirror_mask = (
                mirror_mask
                & (consensus >= self.view_threshold)
                & (visible_count >= self.min_views)
            )

        mirror_rank_score = detached_probability * (consensus if self.use_consensus else 1.0)
        mirror_mask = self._limit_mirror_candidates(mirror_mask, mirror_rank_score, sparse_target)
        uncertain_mask = unlabeled_mask & ~reliable_negative_mask & ~mirror_mask

        positive_element = (probability - gaussian_target).pow(2)
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

        loss_positive = self._masked_mean(positive_element, positive_mask)
        loss_negative = self._masked_mean(negative_element, reliable_negative_mask)
        loss_mirror = self._masked_mean(mirror_element, mirror_mask)
        effective_brl_weight = self.current_brl_weight
        loss = (
            self.positive_weight * loss_positive
            + self.negative_weight * loss_negative
            + effective_brl_weight * loss_mirror
        )

        self.last_stats = {
            "loss_positive": float(loss_positive.detach()),
            "loss_negative": float(loss_negative.detach()),
            "loss_mirror": float(loss_mirror.detach()),
            "brl_weight": float(effective_brl_weight),
            "positive_cells": float(positive_mask.sum().detach()),
            "ignore_cells": float(ignore_mask.sum().detach()),
            "easy_negative_cells": float(easy_negative_mask.sum().detach()),
            "hard_negative_cells": float(hard_negative_mask.sum().detach()),
            "reliable_negative_cells": float(reliable_negative_mask.sum().detach()),
            "mirror_cells": float(mirror_mask.sum().detach()),
            "uncertain_cells": float(uncertain_mask.sum().detach()),
            "visible_cells": float(valid_bev.sum().detach()),
            "max_bev_probability": float(probability.detach().max()),
            "max_view_consensus": float(consensus.detach().max()),
        }
        return loss
