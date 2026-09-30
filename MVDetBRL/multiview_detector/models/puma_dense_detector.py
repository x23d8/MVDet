"""Dense high-recall stage of PUMA-MV.

The model preserves the ``(map_result, per_view_results)`` interface expected
by the existing MVDet trainer while replacing fixed camera concatenation with
multi-height, uncertainty-gated set fusion.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from multiview_detector.models.puma_mv import (
    MultiHeightFeatureSampler,
    UncertaintyGatedSetFusion,
)
from multiview_detector.models.puma_fpn import ResNet18FPNEncoder


def _group_count(channels, maximum=8):
    """Choose the largest valid GroupNorm group count up to ``maximum``."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class PUMADenseDetector(nn.Module):
    def __init__(
        self,
        dataset,
        feature_channels=16,
        fused_channels=48,
        heights_m=(0.0, 0.45, 0.95, 1.45, 1.75),
        world_unit_to_m=None,
        pretrained=False,
        parallel_view_encoding=False,
    ):
        super().__init__()
        if world_unit_to_m is None:
            dataset_name = str(getattr(dataset.base, "__name__", "")).lower()
            if dataset_name == "wildtrack":
                world_unit_to_m = 0.01
            elif dataset_name == "multiviewx":
                world_unit_to_m = 1.0
            else:
                raise ValueError(
                    "world_unit_to_m is required for datasets other than Wildtrack/MultiviewX"
                )
        if world_unit_to_m <= 0:
            raise ValueError("world_unit_to_m must be positive")
        self.world_unit_to_m = float(world_unit_to_m)
        self.num_cam = dataset.num_cam
        self.img_shape = tuple(dataset.img_shape)
        self.reducedgrid_shape = tuple(dataset.reducedgrid_shape)
        self.grid_reduce = int(dataset.grid_reduce)
        self.parallel_view_encoding = bool(parallel_view_encoding)

        self.backbone = ResNet18FPNEncoder(feature_channels, pretrained=pretrained)
        self.image_head = nn.Sequential(
            nn.Conv2d(feature_channels, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 2, kernel_size=1, bias=False),
        )
        self.image_uncertainty = nn.Conv2d(feature_channels, 1, kernel_size=1)

        self.sampler = MultiHeightFeatureSampler(
            heights_m=heights_m, image_shape=self.img_shape
        )
        self.fusion = UncertaintyGatedSetFusion(
            feature_channels,
            fused_channels,
            num_heights=len(heights_m),
            hidden_channels=max(32, feature_channels),
            num_scales=3,
        )
        self.map_head = nn.Sequential(
            nn.Conv2d(fused_channels + 2, fused_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(fused_channels), fused_channels),
            nn.GELU(),
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(fused_channels, 1, kernel_size=1, bias=False),
        )

        projection = self._metric_projection_matrices(dataset, self.world_unit_to_m)
        bev_xy = self._metric_bev_grid(dataset, self.world_unit_to_m)
        coord_map = self._coordinate_map(self.reducedgrid_shape)
        self.register_buffer("projection_matrices", projection)
        self.register_buffer("bev_xy_m", bev_xy)
        self.register_buffer("coord_map", coord_map)

    @staticmethod
    def _metric_projection_matrices(dataset, world_unit_to_m):
        # Dataset calibration consumes its native world unit (centimetres for
        # Wildtrack/MultiviewX). Convert metric XYZ before camera projection.
        metric_to_native = np.diag(
            [1.0 / world_unit_to_m, 1.0 / world_unit_to_m, 1.0 / world_unit_to_m, 1.0]
        )
        matrices = []
        for camera in range(dataset.num_cam):
            native_projection = (
                dataset.base.intrinsic_matrices[camera]
                @ dataset.base.extrinsic_matrices[camera]
            )
            matrices.append(torch.from_numpy(native_projection @ metric_to_native).float())
        return torch.stack(matrices)

    @staticmethod
    def _metric_bev_grid(dataset, world_unit_to_m):
        height, width = dataset.reducedgrid_shape
        row, col = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        if dataset.base.indexing == "xy":
            grid_x = col * dataset.grid_reduce
            grid_y = row * dataset.grid_reduce
        else:
            grid_x = row * dataset.grid_reduce
            grid_y = col * dataset.grid_reduce
        world = dataset.base.get_worldcoord_from_worldgrid(
            np.stack([grid_x, grid_y], axis=0)
        )
        world_xy = np.moveaxis(world, 0, -1) * world_unit_to_m
        return torch.from_numpy(np.asarray(world_xy, dtype=np.float32))

    @staticmethod
    def _coordinate_map(shape):
        height, width = shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)

    def _encode_and_fuse(self, images, camera_mask=None):
        if images.ndim != 5:
            raise ValueError("images must have shape [B,N,C,H,W]")
        batch, num_views = images.shape[:2]
        if num_views != self.num_cam:
            raise ValueError(f"expected {self.num_cam} cameras, received {num_views}")
        if camera_mask is None:
            camera_mask = torch.ones(
                batch, num_views, dtype=torch.bool, device=images.device
            )
        else:
            camera_mask = camera_mask.to(device=images.device, dtype=torch.bool)
            if camera_mask.shape != (batch, num_views):
                raise ValueError("camera_mask must have shape [B,N]")
            if (~camera_mask).all(dim=1).any():
                raise ValueError("each sample must retain at least one camera")

        if self.parallel_view_encoding:
            flat_images = images.reshape(batch * num_views, *images.shape[2:])
            flat_pyramid = self.backbone(flat_images)
            pyramid_features = [
                feature.reshape(batch, num_views, *feature.shape[1:])
                for feature in flat_pyramid
            ]
            flat_image_result = self.image_head(flat_pyramid[0]).reshape(
                batch, num_views, 2, *flat_pyramid[0].shape[-2:]
            )
            image_results = [flat_image_result[:, camera] for camera in range(num_views)]
            pyramid_uncertainty = [
                self.image_uncertainty(feature).reshape(
                    batch, num_views, 1, *feature.shape[-2:]
                )
                for feature in flat_pyramid
            ]
        else:
            pyramid_features = [[], [], []]
            pyramid_uncertainty = [[], [], []]
            image_results = []
            # Sequential view encoding minimizes temporary convolution memory
            # on small GPUs. Parallel mode is substantially faster on T4/A100.
            for camera in range(num_views):
                features = self.backbone(images[:, camera])
                image_results.append(self.image_head(features[0]))
                for scale, feature in enumerate(features):
                    pyramid_features[scale].append(feature)
                    pyramid_uncertainty[scale].append(self.image_uncertainty(feature))

            pyramid_features = [
                torch.stack(features, dim=1) for features in pyramid_features
            ]
            pyramid_uncertainty = [
                torch.stack(features, dim=1) for features in pyramid_uncertainty
            ]
        scale_samples, scale_valid, scale_uncertainty = [], [], []
        for features, uncertainty in zip(pyramid_features, pyramid_uncertainty):
            # Feature and uncertainty share identical projection coordinates;
            # sampling them together avoids a second grid_sample per scale.
            joint = torch.cat([features, uncertainty], dim=2)
            joint_sampled, valid = self.sampler(
                joint, self.projection_matrices, self.bev_xy_m
            )
            scale_samples.append(joint_sampled[:, :, :, :-1])
            scale_valid.append(valid)
            scale_uncertainty.append(joint_sampled[:, :, :, -1:])
        sampled = torch.stack(scale_samples, dim=3)
        valid = torch.stack(scale_valid, dim=3)
        valid = valid & camera_mask[:, :, None, None, None, None, None]
        sampled_uncertainty = torch.stack(scale_uncertainty, dim=3)
        sampled_uncertainty = torch.nn.functional.softplus(sampled_uncertainty)
        fused, fusion_weights = self.fusion(
            sampled, valid, uncertainty=sampled_uncertainty
        )
        coordinate = self.coord_map.expand(batch, -1, -1, -1)
        map_result = self.map_head(torch.cat([fused, coordinate], dim=1))

        auxiliary = {
            "fusion_weights": fusion_weights,
            "valid_samples": valid,
            "sample_uncertainty": sampled_uncertainty,
            "camera_mask": camera_mask,
        }
        return map_result, image_results, pyramid_features, fused, auxiliary

    def forward(self, images, return_aux=False, camera_mask=None):
        map_result, image_results, _, _, auxiliary = self._encode_and_fuse(
            images, camera_mask=camera_mask
        )

        if return_aux:
            return map_result, image_results, auxiliary
        return map_result, image_results
