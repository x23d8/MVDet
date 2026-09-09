#!/usr/bin/env python3
"""Synthetic end-to-end MVDet + AdaptiveBRL forward/backward smoke test."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# Direct ``python tools/...py`` execution otherwise puts only tools/ on sys.path.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from multiview_detector.loss import AdaptiveBRLLoss, MultiViewAdaptiveBRLLoss
from multiview_detector.models.persp_trans_detector import PerspTransDetector


def fake_dataset(num_cameras=2):
    extrinsic = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    base = SimpleNamespace(
        intrinsic_matrices=[np.eye(3) for _ in range(num_cameras)],
        extrinsic_matrices=[extrinsic for _ in range(num_cameras)],
        worldgrid2worldcoord_mat=np.eye(3),
    )
    return SimpleNamespace(
        num_cam=num_cameras,
        img_shape=[32, 32],
        reducedgrid_shape=[8, 8],
        img_reduce=4,
        grid_reduce=4,
        base=base,
    )


def gaussian_kernel(channels=1):
    axis = torch.arange(3) - 1
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    gaussian = torch.exp(-(xx.square() + yy.square()) / 2.0)
    kernel = torch.zeros(channels, channels, 3, 3)
    for channel in range(channels):
        kernel[channel, channel] = gaussian
    return kernel


def main():
    torch.manual_seed(7)
    dataset = fake_dataset()
    model = PerspTransDetector(dataset).train()
    images = torch.randn(1, dataset.num_cam, 3, *dataset.img_shape)
    map_logits, view_logits = model(images)

    map_target = torch.zeros_like(map_logits, device='cpu')
    map_target[0, 0, 2, 2] = 1
    view_targets = []
    for logits in view_logits:
        target = torch.zeros_like(logits, device='cpu')
        target[0, :, 2, 2] = 1
        view_targets.append(target)

    map_criterion = MultiViewAdaptiveBRLLoss(
        annotation_probability=0.5,
        warmup_epochs=0,
        ramp_epochs=0,
    ).to(model.fusion_device)
    view_criterion = AdaptiveBRLLoss(
        annotation_probability=0.5,
        warmup_epochs=0,
        ramp_epochs=0,
    ).to(model.fusion_device)
    map_criterion.set_epoch(1)
    view_criterion.set_epoch(1)

    loss = map_criterion(
        map_logits,
        map_target,
        gaussian_kernel(),
        view_logits=view_logits,
        projection_matrices=model.proj_mats,
    )
    for logits, target in zip(view_logits, view_targets):
        loss = loss + view_criterion(logits, target, gaussian_kernel(2)) / len(view_logits)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise RuntimeError('missing or non-finite model gradients')
    summary = {
        'front_device': str(model.front_device),
        'fusion_device': str(model.fusion_device),
        'map_shape': list(map_logits.shape),
        'view_shapes': [list(logits.shape) for logits in view_logits],
        'loss': float(loss.detach()),
        'parameters_with_gradient': len(gradients),
        'bev_stats': map_criterion.last_stats,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
