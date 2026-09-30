#!/usr/bin/env python3
"""One real Wildtrack AMP train step for dataset/geometry integration checks."""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multiview_detector.datasets.Wildtrack import Wildtrack
from multiview_detector.datasets.frameDataset import frameDataset
from multiview_detector.loss.pu_gaussian_mse import PUGaussianMSE
from multiview_detector.loss.pu_query_loss import PUQuerySetLoss
from multiview_detector.loss.camera_drop_consistency import CameraDropConsistencyLoss
from multiview_detector.models.puma_hybrid_detector import PUMAHybridDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--gt-path", type=Path, default=None)
    parser.add_argument("--query-loss-weight", type=float, default=0.02)
    parser.add_argument("--feature-channels", type=int, default=16)
    parser.add_argument("--fused-channels", type=int, default=48)
    parser.add_argument("--query-channels", type=int, default=48)
    parser.add_argument("--parallel-view-encoding", action="store_true")
    parser.add_argument("--consistency-pass", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    transform = T.Compose([
        T.Resize([720, 1280]),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = frameDataset(
        Wildtrack(str(args.root)), train=True, transform=transform,
        grid_reduce=4, frame_range=(0, 1),
        gt_fpath=str(args.gt_path) if args.gt_path else None,
        force_download=args.gt_path is not None,
    )
    images, map_target, image_targets, frame = dataset[0]
    images = images.unsqueeze(0).cuda()
    map_target = map_target.unsqueeze(0).cuda()
    image_targets = [target.unsqueeze(0).cuda() for target in image_targets]
    model = PUMAHybridDetector(
        dataset,
        feature_channels=args.feature_channels,
        fused_channels=args.fused_channels,
        num_queries=64,
        query_hidden_channels=args.query_channels,
        query_layers=2,
        parallel_view_encoding=args.parallel_view_encoding,
    ).cuda().train()
    heatmap_loss = PUGaussianMSE(annotation_propensity=1.0).cuda()
    query_loss = PUQuerySetLoss(
        annotation_propensity=1.0,
        coordinate_weight=1.0,
    ).cuda()
    torch.cuda.reset_peak_memory_stats()
    teacher_map = None
    camera_mask = None
    if args.consistency_pass:
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            teacher_map, _ = model(images)
        model.train()
        camera_mask = torch.ones(1, dataset.num_cam, dtype=torch.bool, device="cuda")
        camera_mask[:, 0] = False
    with torch.autocast("cuda", dtype=torch.float16):
        prediction, image_predictions, auxiliary = model(
            images, return_aux=True, camera_mask=camera_mask
        )
        heatmap_component = heatmap_loss(prediction, map_target, dataset.map_kernel)
        image_component = prediction.new_zeros(())
        for image_prediction, image_target in zip(image_predictions, image_targets):
            image_component = image_component + heatmap_loss(
                image_prediction, image_target, dataset.img_kernel
            ) / dataset.num_cam
        query_component = query_loss(
            auxiliary["queries"], map_target, model.bev_xy_m
        )
        loss = (
            heatmap_component + image_component
            + args.query_loss_weight * query_component
        )
        consistency_component = prediction.new_zeros(())
        if teacher_map is not None:
            consistency_component = CameraDropConsistencyLoss()(prediction, teacher_map)
            loss = loss + 0.02 * consistency_component
    loss.backward()
    valid = auxiliary["valid_samples"].any(dim=(2, 3, 4, 5, 6))[0]
    if not valid.all():
        raise RuntimeError(f"projection produced no valid BEV samples for cameras: {(~valid).nonzero()}")
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise RuntimeError("non-finite prediction/loss")
    print({
        "frame": int(frame),
        "images": tuple(images.shape),
        "map": tuple(prediction.shape),
        "observed_people": int((map_target > 0).sum()),
        "loss": float(loss.detach()),
        "heatmap_loss": float(heatmap_component.detach()),
        "image_loss": float(image_component.detach()),
        "query_loss_unweighted": float(query_component.detach()),
        "query_loss_weight": args.query_loss_weight,
        "consistency_loss": float(consistency_component.detach()),
        "parallel_view_encoding": args.parallel_view_encoding,
        "channels": [args.feature_channels, args.fused_channels, args.query_channels],
        "query_components": query_loss.last_components,
        "valid_camera_samples": valid.tolist(),
        "peak_memory_mib": int(torch.cuda.max_memory_allocated() / (1024 ** 2)),
    })


if __name__ == "__main__":
    main()
