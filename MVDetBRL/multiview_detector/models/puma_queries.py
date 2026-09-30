"""Sparse continuous query blocks for PUMA-MV.

The dense branch supplies high-recall proposals.  These blocks turn local
maxima into metric BEV queries and re-sample the original per-view feature
maps on a small 3-D cylinder around every proposal.  Camera and cylinder
samples are fused as a set, so camera order does not carry semantic meaning.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class DensePeakQueryInitializer(nn.Module):
    """Create metric queries from spatially separated dense local maxima."""

    def __init__(self, num_queries=64, suppression_kernel=5):
        super().__init__()
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if suppression_kernel <= 0 or suppression_kernel % 2 == 0:
            raise ValueError("suppression_kernel must be a positive odd number")
        self.num_queries = int(num_queries)
        self.suppression_kernel = int(suppression_kernel)

    def forward(self, dense_score, bev_feature, bev_xy_m):
        if dense_score.ndim != 4 or dense_score.shape[1] != 1:
            raise ValueError("dense_score must have shape [B,1,H,W]")
        if bev_feature.ndim != 4 or bev_feature.shape[0] != dense_score.shape[0]:
            raise ValueError("bev_feature must have shape [B,C,H,W]")
        if bev_feature.shape[-2:] != dense_score.shape[-2:]:
            raise ValueError("dense_score and bev_feature spatial shapes must match")
        batch, _, height, width = dense_score.shape
        if self.num_queries > height * width:
            raise ValueError("num_queries cannot exceed the number of BEV cells")
        if bev_xy_m.ndim == 3:
            bev_xy_m = bev_xy_m.unsqueeze(0).expand(batch, -1, -1, -1)
        if bev_xy_m.shape != (batch, height, width, 2):
            raise ValueError("bev_xy_m must have shape [H,W,2] or [B,H,W,2]")

        pooled = F.max_pool2d(
            dense_score,
            kernel_size=self.suppression_kernel,
            stride=1,
            padding=self.suppression_kernel // 2,
        )
        local_score = dense_score.masked_fill(
            dense_score < pooled, torch.finfo(dense_score.dtype).min
        )
        scores, indices = local_score.flatten(2).topk(self.num_queries, dim=-1)
        scores, indices = scores.squeeze(1), indices.squeeze(1)

        flat_xy = bev_xy_m.reshape(batch, height * width, 2)
        xy = flat_xy.gather(1, indices.unsqueeze(-1).expand(-1, -1, 2))
        channels = bev_feature.shape[1]
        flat_feature = bev_feature.flatten(2).transpose(1, 2)
        features = flat_feature.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, channels)
        )
        return xy, features, scores, indices


class CylindricalQueryRefiner(nn.Module):
    """Refine metric BEV points using per-view samples on a 3-D cylinder.

    The output coordinates stay continuous in metres.  Invalid projections
    are masked and a learned null token lets a query ignore all views.
    """

    def __init__(
        self,
        image_channels,
        query_channels,
        hidden_channels=64,
        heights_m=(0.0, 0.9, 1.7),
        radius_m=0.25,
        radial_samples=4,
        num_layers=2,
        max_offset_m=0.5,
        local_radius_m=1.5,
        image_shape=(1080, 1920),
        num_scales=1,
        initial_sigma_m=0.35,
    ):
        super().__init__()
        if min(image_channels, query_channels, hidden_channels, radial_samples,
               num_layers, num_scales) <= 0:
            raise ValueError("channel/sample/layer counts must be positive")
        if (radius_m < 0 or max_offset_m <= 0 or local_radius_m <= 0
                or initial_sigma_m <= 0):
            raise ValueError("metric radii and max_offset_m must be valid")
        heights = torch.as_tensor(heights_m, dtype=torch.float32)
        if heights.ndim != 1 or heights.numel() == 0:
            raise ValueError("heights_m must be a non-empty sequence")

        angles = torch.arange(radial_samples, dtype=torch.float32) * (
            2.0 * math.pi / radial_samples
        )
        ring = radius_m * torch.stack([angles.cos(), angles.sin()], dim=-1)
        ring = torch.cat([torch.zeros(1, 2), ring], dim=0)
        xy_offsets = ring.repeat(heights.numel(), 1)
        z_offsets = heights.repeat_interleave(ring.shape[0])
        self.register_buffer("sample_xy_offsets_m", xy_offsets)
        self.register_buffer("sample_z_m", z_offsets)

        self.image_shape = tuple(int(value) for value in image_shape)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.num_scales = int(num_scales)
        self.max_offset_m = float(max_offset_m)
        self.local_radius_m = float(local_radius_m)

        self.query_input = nn.Linear(query_channels + 1, hidden_channels)
        self.sample_value = nn.Linear(image_channels, hidden_channels)
        self.sample_key = nn.Linear(image_channels, hidden_channels)
        self.sample_embedding = nn.Parameter(
            torch.zeros(self.sample_z_m.numel(), image_channels)
        )
        self.scale_embedding = nn.Parameter(torch.zeros(num_scales, image_channels))
        nn.init.normal_(self.sample_embedding, std=0.02)
        nn.init.normal_(self.scale_embedding, std=0.02)
        self.null_key = nn.Parameter(torch.zeros(1, 1, 1, hidden_channels))
        self.null_value = nn.Parameter(torch.zeros(1, 1, 1, hidden_channels))

        self.cross_norms = nn.ModuleList(
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        )
        self.local_norms = nn.ModuleList(
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        )
        self.local_q = nn.ModuleList(
            nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)
        )
        self.local_k = nn.ModuleList(
            nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)
        )
        self.local_v = nn.ModuleList(
            nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)
        )
        self.offset_heads = nn.ModuleList(
            nn.Linear(hidden_channels, 2) for _ in range(num_layers)
        )
        self.objectness = nn.Linear(hidden_channels, 1)
        self.propensity = nn.Linear(hidden_channels, 1)
        self.log_sigma = nn.Linear(hidden_channels, 2)
        nn.init.constant_(self.log_sigma.bias, math.log(initial_sigma_m))

    def _sample_single_scale(self, image_features, projection, xy_m):
        batch, num_views, channels, _, _ = image_features.shape
        num_queries = xy_m.shape[1]
        num_samples = self.sample_z_m.numel()
        sample_xy = xy_m[:, :, None] + self.sample_xy_offsets_m.to(xy_m)[None, None]
        z = self.sample_z_m.to(xy_m).view(1, 1, num_samples, 1).expand(
            batch, num_queries, -1, -1
        )
        world = torch.cat([sample_xy, z, torch.ones_like(z)], dim=-1)
        if projection.ndim == 3:
            projection = projection.unsqueeze(0).expand(batch, -1, -1, -1)
        if projection.shape != (batch, num_views, 3, 4):
            raise ValueError("projection must have shape [N,3,4] or [B,N,3,4]")

        with torch.autocast(device_type=image_features.device.type, enabled=False):
            image_h = torch.einsum(
                "bnij,bksj->bnksi", projection.float(), world.float()
            )
            depth = image_h[..., 2]
            safe_depth = depth.clamp_min(1e-6)
            u = image_h[..., 0] / safe_depth
            v = image_h[..., 1] / safe_depth
            image_height, image_width = self.image_shape
            grid_x = 2.0 * u / (image_width - 1) - 1.0
            grid_y = 2.0 * v / (image_height - 1) - 1.0
            valid = (
                (depth > 1e-6)
                & torch.isfinite(depth)
                & torch.isfinite(grid_x)
                & torch.isfinite(grid_y)
                & (grid_x >= -1.0)
                & (grid_x <= 1.0)
                & (grid_y >= -1.0)
                & (grid_y <= 1.0)
            )
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = torch.where(valid.unsqueeze(-1), grid, torch.zeros_like(grid)).reshape(
            batch * num_views, num_queries * num_samples, 1, 2
        )
        flat_features = image_features.reshape(
            batch * num_views, channels, *image_features.shape[-2:]
        )
        sampled = F.grid_sample(
            flat_features, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(
            batch, num_views, num_queries, num_samples, channels
        )
        return sampled, valid

    def _sample_cylinders(self, image_features, projection, xy_m, camera_mask=None):
        if torch.is_tensor(image_features):
            feature_scales = [image_features]
        else:
            feature_scales = list(image_features)
        if len(feature_scales) != self.num_scales:
            raise ValueError(
                f"expected {self.num_scales} image scales, received {len(feature_scales)}"
            )
        sampled_scales, valid_scales = [], []
        for scale, features in enumerate(feature_scales):
            sampled, valid = self._sample_single_scale(features, projection, xy_m)
            sampled = sampled + self.sample_embedding.to(sampled)[None, None, None]
            sampled = sampled + self.scale_embedding[scale].to(sampled).view(1, 1, 1, 1, -1)
            sampled_scales.append(sampled)
            valid_scales.append(valid)
        # Treat scale like another set element; it is not folded into channels.
        sampled = torch.cat(sampled_scales, dim=3)
        valid = torch.cat(valid_scales, dim=3)
        if camera_mask is not None:
            camera_mask = camera_mask.to(device=valid.device, dtype=torch.bool)
            if camera_mask.shape != valid.shape[:2]:
                raise ValueError("camera_mask must have shape [B,N]")
            valid = valid & camera_mask[:, :, None, None]
        return sampled, valid

    def _cross_view_update(self, query, sampled, valid):
        batch, num_views, num_queries, num_samples, channels = sampled.shape
        tokens = sampled.permute(0, 2, 1, 3, 4).reshape(
            batch, num_queries, num_views * num_samples, channels
        )
        token_valid = valid.permute(0, 2, 1, 3).reshape(
            batch, num_queries, num_views * num_samples
        )
        keys = self.sample_key(tokens)
        values = self.sample_value(tokens)
        logits = (keys * query.unsqueeze(2)).sum(dim=-1) / math.sqrt(self.hidden_channels)
        logits = logits.masked_fill(~token_valid, torch.finfo(logits.dtype).min)
        null_key = self.null_key.expand(batch, num_queries, -1, -1)
        null_value = self.null_value.expand(batch, num_queries, -1, -1)
        null_logit = (null_key * query.unsqueeze(2)).sum(dim=-1) / math.sqrt(
            self.hidden_channels
        )
        weights = torch.softmax(torch.cat([logits, null_logit], dim=2), dim=2)
        all_values = torch.cat([values, null_value], dim=2)
        return (weights.unsqueeze(-1) * all_values).sum(dim=2), weights

    def _local_update(self, layer, query, xy_m):
        q = self.local_q[layer](query)
        k = self.local_k[layer](query)
        v = self.local_v[layer](query)
        logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.hidden_channels)
        distance = torch.cdist(xy_m, xy_m)
        logits = logits.masked_fill(distance > self.local_radius_m, torch.finfo(logits.dtype).min)
        return torch.matmul(torch.softmax(logits, dim=-1), v)

    def forward(self, image_features, projection, initial_xy_m, initial_features,
                initial_scores, camera_mask=None):
        if torch.is_tensor(image_features) and image_features.ndim != 5:
            raise ValueError("image_features must have shape [B,N,C,H,W]")
        if initial_xy_m.ndim != 3 or initial_xy_m.shape[-1] != 2:
            raise ValueError("initial_xy_m must have shape [B,K,2]")
        if initial_features.shape[:2] != initial_xy_m.shape[:2]:
            raise ValueError("initial_features and initial_xy_m must share B,K")
        if initial_scores.shape != initial_xy_m.shape[:2]:
            raise ValueError("initial_scores must have shape [B,K]")

        query = self.query_input(
            torch.cat([initial_features, initial_scores.sigmoid().unsqueeze(-1)], dim=-1)
        )
        xy_m = initial_xy_m
        attention = None
        for layer in range(self.num_layers):
            sampled, valid = self._sample_cylinders(
                image_features, projection, xy_m, camera_mask=camera_mask
            )
            cross, attention = self._cross_view_update(query, sampled, valid)
            query = self.cross_norms[layer](query + cross)
            local = self._local_update(layer, query, xy_m)
            query = self.local_norms[layer](query + local)
            delta = torch.tanh(self.offset_heads[layer](query)) * self.max_offset_m
            xy_m = xy_m + delta

        return {
            "xy_m": xy_m,
            "objectness": self.objectness(query).squeeze(-1),
            "propensity_logit": self.propensity(query).squeeze(-1),
            "log_sigma_m": self.log_sigma(query).clamp(-5.0, 2.0),
            "embedding": query,
            "attention": attention,
            "valid_samples": valid,
        }
