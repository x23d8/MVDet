"""Core geometry and fusion blocks for the proposed PUMA-MV detector.

This module intentionally contains independently testable building blocks. The
end-to-end dense-to-sparse detector will be assembled only after the partial
label and lifting baselines are validated.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MultiHeightFeatureSampler(nn.Module):
    """Pull image features at vertical world samples above each BEV location.

    Args:
        heights_m: World-space sample heights in metres.
        image_shape: Original image ``(height, width)`` used by the projection
            matrices. Normalized coordinates make the sampling independent of
            the feature-map stride.

    Inputs:
        features: ``[B, N, C, Hf, Wf]``.
        projection: ``[B, N, 3, 4]`` mapping homogeneous world XYZ to image
            pixels. ``[N, 3, 4]`` is also accepted and broadcast over B.
        bev_xy_m: ``[Hb, Wb, 2]`` or ``[B, Hb, Wb, 2]`` in world metres.

    Returns:
        samples: ``[B, N, Z, C, Hb, Wb]``.
        valid: ``[B, N, Z, 1, Hb, Wb]``.
    """

    def __init__(self, heights_m=(0.0, 0.45, 0.95, 1.45, 1.75), image_shape=(1080, 1920)):
        super().__init__()
        heights = torch.as_tensor(heights_m, dtype=torch.float32)
        if heights.ndim != 1 or heights.numel() == 0:
            raise ValueError("heights_m must be a non-empty one-dimensional sequence")
        if image_shape[0] <= 1 or image_shape[1] <= 1:
            raise ValueError("image_shape dimensions must be greater than one")
        self.register_buffer("heights_m", heights)
        self.image_shape = tuple(int(value) for value in image_shape)

    def forward(self, features, projection, bev_xy_m):
        if features.ndim != 5:
            raise ValueError("features must have shape [B,N,C,Hf,Wf]")
        batch, num_views, channels, _, _ = features.shape
        if projection.ndim == 3:
            projection = projection.unsqueeze(0).expand(batch, -1, -1, -1)
        if projection.shape != (batch, num_views, 3, 4):
            raise ValueError("projection must have shape [B,N,3,4] or [N,3,4]")
        if bev_xy_m.ndim == 3:
            bev_xy_m = bev_xy_m.unsqueeze(0).expand(batch, -1, -1, -1)
        if bev_xy_m.ndim != 4 or bev_xy_m.shape[0] != batch or bev_xy_m.shape[-1] != 2:
            raise ValueError("bev_xy_m must have shape [Hb,Wb,2] or [B,Hb,Wb,2]")

        bev_height, bev_width = bev_xy_m.shape[1:3]
        num_heights = self.heights_m.numel()
        # Calibration multiplication must remain float32 under AMP. Wildtrack
        # contains K[R|t] coefficients that overflow float16 after converting
        # metric coordinates to native centimetres.
        geometry_dtype = torch.float32
        xy = bev_xy_m.to(device=features.device, dtype=geometry_dtype)
        xy = xy[:, None].expand(-1, num_heights, -1, -1, -1)
        z = self.heights_m.to(device=features.device, dtype=geometry_dtype).view(
            1, num_heights, 1, 1, 1
        ).expand(
            batch, -1, bev_height, bev_width, -1
        )
        ones = torch.ones_like(z)
        world = torch.cat([xy, z, ones], dim=-1)

        # [B,N,Z,Hb,Wb,3]
        with torch.autocast(device_type=features.device.type, enabled=False):
            image_h = torch.einsum(
                "bnij,bzhwj->bnzhwi",
                projection.to(device=features.device, dtype=geometry_dtype),
                world,
            )
            depth = image_h[..., 2]
            safe_depth = depth.clamp_min(1e-6)
            u = image_h[..., 0] / safe_depth
            v = image_h[..., 1] / safe_depth
            image_height, image_width = self.image_shape
            grid_x = 2.0 * u / (image_width - 1) - 1.0
            grid_y = 2.0 * v / (image_height - 1) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1)
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
        # grid_sample may propagate NaN even if its output is multiplied by a
        # false validity mask afterwards. Redirect invalid coordinates to a
        # finite in-bounds location before sampling, then zero them by mask.
        grid = torch.where(valid.unsqueeze(-1), grid, torch.zeros_like(grid))

        flat_features = features.reshape(batch * num_views, channels, *features.shape[-2:])
        flat_grid = grid.to(dtype=features.dtype).reshape(
            batch * num_views, num_heights * bev_height, bev_width, 2
        )
        sampled = F.grid_sample(
            flat_features,
            flat_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.reshape(
            batch, num_views, channels, num_heights, bev_height, bev_width
        ).permute(0, 1, 3, 2, 4, 5)
        valid = valid.unsqueeze(3)
        sampled = sampled * valid.to(sampled.dtype)
        return sampled, valid


class UncertaintyGatedSetFusion(nn.Module):
    """Permutation-invariant fusion across cameras and height samples.

    A shared value/score transform is applied to every token. Invalid tokens
    receive zero probability. A learned null token lets the model reject all
    camera evidence at a location instead of selecting the least-bad view.
    """

    def __init__(self, in_channels, out_channels, num_heights, hidden_channels=64,
                 num_scales=1):
        super().__init__()
        if min(in_channels, out_channels, num_heights, hidden_channels, num_scales) <= 0:
            raise ValueError("channel, height, and scale counts must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_heights = int(num_heights)
        self.num_scales = int(num_scales)
        self.height_embedding = nn.Parameter(torch.zeros(num_heights, in_channels))
        self.scale_embedding = nn.Parameter(torch.zeros(num_scales, in_channels))
        nn.init.normal_(self.height_embedding, std=0.02)
        nn.init.normal_(self.scale_embedding, std=0.02)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.score = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.uncertainty_scale = nn.Parameter(torch.tensor(1.0))
        self.null_value = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.null_score = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, samples, valid, uncertainty=None):
        if samples.ndim == 6:
            samples = samples.unsqueeze(3)
            valid = valid.unsqueeze(3)
            if uncertainty is not None:
                uncertainty = uncertainty.unsqueeze(3)
        if samples.ndim != 7:
            raise ValueError("samples must have shape [B,N,Z,C,H,W] or [B,N,Z,L,C,H,W]")
        batch, num_views, num_heights, num_scales, channels, height, width = samples.shape
        if (channels != self.in_channels or num_heights != self.num_heights
                or num_scales != self.num_scales):
            raise ValueError("samples do not match configured channels/heights/scales")
        expected_valid = (batch, num_views, num_heights, num_scales, 1, height, width)
        if valid.shape != expected_valid:
            raise ValueError(f"valid must have shape {expected_valid}")
        if uncertainty is not None and uncertainty.shape != expected_valid:
            raise ValueError(f"uncertainty must have shape {expected_valid}")

        tokens = samples + self.height_embedding.view(
            1, 1, num_heights, 1, channels, 1, 1
        ) + self.scale_embedding.view(1, 1, 1, num_scales, channels, 1, 1)
        num_tokens = num_views * num_heights * num_scales
        flat = tokens.reshape(batch * num_tokens, channels, height, width)
        values = self.value(flat).reshape(
            batch, num_tokens, self.out_channels, height, width
        )
        logits = self.score(flat).reshape(batch, num_tokens, 1, height, width)
        if uncertainty is not None:
            scale = F.softplus(self.uncertainty_scale)
            logits = logits - scale * uncertainty.reshape_as(logits).clamp_min(0.0)
        token_valid = valid.reshape(batch, num_tokens, 1, height, width)
        logits = logits.masked_fill(~token_valid, torch.finfo(logits.dtype).min)

        null_score = self.null_score.expand(batch, -1, height, width).unsqueeze(1)
        all_logits = torch.cat([logits, null_score], dim=1)
        weights = torch.softmax(all_logits, dim=1)
        token_weights, null_weight = weights[:, :-1], weights[:, -1]
        fused = (token_weights * values).sum(dim=1)
        fused = fused + null_weight * self.null_value
        return fused, weights
