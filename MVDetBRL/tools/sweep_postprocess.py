#!/usr/bin/env python3
"""Tune heatmap threshold and metric point-NMS on a validation score cache."""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multiview_detector.evaluation.pyeval.evaluateDetection import evaluateDetection_py
from multiview_detector.utils.nms import nms


def detections_from_maps(maps, frames, threshold, radius_grid, grid_reduce, indexing):
    detections = []
    for score_map, frame in zip(maps, frames):
        rows_cols = np.argwhere(score_map > threshold)
        if rows_cols.size == 0:
            continue
        scores = torch.from_numpy(score_map[rows_cols[:, 0], rows_cols[:, 1]])
        if indexing == "xy":
            coordinates = rows_cols[:, [1, 0]]
        else:
            coordinates = rows_cols
        positions = torch.from_numpy(coordinates).float() * grid_reduce
        keep, count = nms(positions, scores, radius_grid, np.inf)
        for position in positions[keep[:count]].numpy():
            detections.append([int(frame), position[0], position[1]])
    return np.asarray(detections, dtype=np.float64).reshape(-1, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("score_cache", type=Path)
    parser.add_argument("gt", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--nms-radii-m", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--grid-cell-m", type=float, default=0.025)
    parser.add_argument("--dataset", default="Wildtrack")
    parser.add_argument("--output", type=Path, default=Path("postprocess_sweep.csv"))
    args = parser.parse_args()
    cache = np.load(args.score_cache)
    maps, frames = cache["maps"], cache["frames"]
    grid_reduce = float(cache["grid_reduce"])
    indexing = str(cache["indexing"])
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        result_path = Path(directory) / "result.txt"
        for threshold in args.thresholds:
            for radius_m in args.nms_radii_m:
                detections = detections_from_maps(
                    maps, frames, threshold, radius_m / args.grid_cell_m,
                    grid_reduce, indexing,
                )
                np.savetxt(result_path, detections, fmt="%.8f")
                recall, precision, moda, modp = evaluateDetection_py(
                    result_path, args.gt, args.dataset, frames=frames
                )
                rows.append({
                    "threshold": threshold, "nms_radius_m": radius_m,
                    "moda": moda, "modp": modp,
                    "precision": precision, "recall": recall,
                })
    rows.sort(key=lambda row: (row["moda"], row["modp"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Best validation setting: {rows[0]}")


if __name__ == "__main__":
    main()
