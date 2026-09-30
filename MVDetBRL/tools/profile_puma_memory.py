#!/usr/bin/env python3
"""Profile one synthetic PUMA-MV train step at dataset-scale tensor sizes."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multiview_detector.models.puma_hybrid_detector import PUMAHybridDetector


def fake_dataset(num_cameras, dataset_name, bev_height, bev_width):
    base = SimpleNamespace(indexing="xy", __name__=dataset_name)
    intrinsic = np.array(
        [[400.0, 0.0, 640.0], [0.0, 400.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    extrinsic = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    base.intrinsic_matrices = [intrinsic.copy() for _ in range(num_cameras)]
    base.extrinsic_matrices = [extrinsic.copy() for _ in range(num_cameras)]
    base.get_worldcoord_from_worldgrid = lambda grid: grid.astype(np.float32)
    return SimpleNamespace(
        num_cam=num_cameras,
        img_shape=[1080, 1920],
        reducedgrid_shape=[bev_height, bev_width],
        grid_reduce=4,
        base=base,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-cameras", type=int, default=7)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--bev-height", type=int, default=120)
    parser.add_argument("--bev-width", type=int, default=360)
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--fused-channels", type=int, default=64)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--parallel-view-encoding", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory profiling")

    dataset = fake_dataset(
        args.num_cameras, "Wildtrack", args.bev_height, args.bev_width
    )
    model = PUMAHybridDetector(
        dataset,
        feature_channels=args.feature_channels,
        fused_channels=args.fused_channels,
        num_queries=args.queries,
        query_hidden_channels=args.fused_channels,
        query_layers=2,
        parallel_view_encoding=args.parallel_view_encoding,
    ).cuda().train()
    images = torch.randn(
        args.batch_size,
        args.num_cameras,
        3,
        args.height,
        args.width,
        device="cuda",
    )
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        prediction, _ = model(images)
        loss = prediction.square().mean()
    loss.backward()
    allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    print(
        f"output={tuple(prediction.shape)} peak_allocated={allocated:.0f} MiB "
        f"peak_reserved={reserved:.0f} MiB"
    )


if __name__ == "__main__":
    main()
