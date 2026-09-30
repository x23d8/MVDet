"""End-to-end dense-to-sparse PUMA-MV detector."""

from __future__ import annotations

import torch
from torch import nn

from multiview_detector.models.puma_dense_detector import PUMADenseDetector
from multiview_detector.models.puma_queries import (
    CylindricalQueryRefiner,
    DensePeakQueryInitializer,
)


class PUMAHybridDetector(PUMADenseDetector):
    """Dense recall followed by continuous, image-grounded query refinement.

    Refined queries are rendered back into the heatmap with an initially small
    residual gate.  This keeps the legacy trainer/evaluator operational while
    allowing gradients from the map objective to train query coordinates and
    objectness.  The query dictionary is available through ``return_aux`` for
    the later set/geometry objectives.
    """

    def __init__(
        self,
        dataset,
        feature_channels=16,
        fused_channels=48,
        heights_m=(0.0, 0.45, 0.95, 1.45, 1.75),
        world_unit_to_m=None,
        num_queries=64,
        query_hidden_channels=48,
        query_layers=2,
        query_sigma_m=0.35,
        pretrained=False,
        parallel_view_encoding=False,
    ):
        super().__init__(
            dataset,
            feature_channels=feature_channels,
            fused_channels=fused_channels,
            heights_m=heights_m,
            world_unit_to_m=world_unit_to_m,
            pretrained=pretrained,
            parallel_view_encoding=parallel_view_encoding,
        )
        if query_sigma_m <= 0:
            raise ValueError("query_sigma_m must be positive")
        self.query_initializer = DensePeakQueryInitializer(num_queries=num_queries)
        self.query_refiner = CylindricalQueryRefiner(
            image_channels=feature_channels,
            query_channels=fused_channels,
            hidden_channels=query_hidden_channels,
            num_layers=query_layers,
            image_shape=self.img_shape,
            num_scales=3,
            initial_sigma_m=query_sigma_m,
        )
        self.query_sigma_m = float(query_sigma_m)
        # Start as an almost pure dense model, then let optimization introduce
        # the sparse residual instead of destabilizing the first iterations.
        self.query_residual_logit = nn.Parameter(torch.tensor(-4.0))

    def _render_queries(self, queries):
        xy = queries["xy_m"]
        confidence = queries["objectness"].sigmoid()
        sigma = queries["log_sigma_m"].exp().clamp(0.05, 1.5)
        grid = self.bev_xy_m.to(xy)[None, None]
        delta = grid - xy[:, :, None, None, :]
        exponent = -0.5 * ((delta / sigma[:, :, None, None, :]) ** 2).sum(dim=-1)
        heatmaps = confidence[:, :, None, None] * exponent.exp()
        return heatmaps.amax(dim=1, keepdim=True)

    def forward(self, images, return_aux=False, camera_mask=None):
        dense_map, image_results, compact, fused, auxiliary = self._encode_and_fuse(
            images, camera_mask=camera_mask
        )
        initial_xy, initial_feature, initial_score, initial_index = self.query_initializer(
            dense_map, fused, self.bev_xy_m
        )
        queries = self.query_refiner(
            compact,
            self.projection_matrices,
            initial_xy,
            initial_feature,
            initial_score,
            camera_mask=camera_mask,
        )
        query_map = self._render_queries(queries)
        residual_gate = self.query_residual_logit.sigmoid()
        map_result = dense_map + residual_gate * query_map
        if return_aux:
            auxiliary.update(
                {
                    "dense_map": dense_map,
                    "query_map": query_map,
                    "query_residual_gate": residual_gate,
                    "query_initial_indices": initial_index,
                    "queries": queries,
                }
            )
            return map_result, image_results, auxiliary
        return map_result, image_results
