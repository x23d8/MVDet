"""Visualize YOLO foot projections on the exact reduced BEV grid used by MVDet."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal


TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parents[1]
MVDET_DIR = REPO_DIR / "MVDetBRL"
if str(MVDET_DIR) not in sys.path:
    sys.path.insert(0, str(MVDET_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from multiview_detector.datasets.Wildtrack import Wildtrack
from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
from yolo_pose_foot_demo import extract_predictions


CAMERA_COLORS = ["#e63946", "#f4a261", "#e9c46a", "#2a9d8f", "#457b9d", "#6a4c93", "#ef476f"]


def parse_args():
    parser = argparse.ArgumentParser(description="Project YOLO foot points to MVDet's Wildtrack BEV grid")
    parser.add_argument("--image-subsets", type=Path, required=True,
                        help="Wildtrack Image_subsets directory containing C1...C7")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo26m-pose.pt")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--grid-reduce", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--kpt-conf", type=float, default=0.35)
    parser.add_argument("--candidate-conf", type=float, default=0.15)
    parser.add_argument("--foot-anchor", choices=["pose", "bbox_bottom", "pose_x_bbox_y"],
                        default="pose_x_bbox_y")
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--merge-radius-m", type=float, default=0.75)
    parser.add_argument("--temporal-singletons", action="store_true",
                        help="rescue a one-camera cluster only when it continues an unmatched prior track")
    parser.add_argument("--temporal-conf", type=float, default=0.65,
                        help="minimum confidence for a temporally supported singleton")
    parser.add_argument("--temporal-radius-m", type=float, default=0.6,
                        help="maximum displacement between adjacent annotated frames")
    parser.add_argument("--inference-batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--show-gt", action="store_true",
                        help="overlay full GT positions for a projection sanity check")
    parser.add_argument("--eval-radius-m", type=float, default=0.5,
                        help="maximum BEV distance for GT matching when --show-gt is enabled")
    return parser.parse_args()


def grid_to_mvdet_cell(base, grid_x, grid_y, grid_reduce):
    """Match frameDataset's row/column convention exactly."""
    if base.indexing == "xy":
        row, col = grid_y / grid_reduce, grid_x / grid_reduce
    else:
        row, col = grid_x / grid_reduce, grid_y / grid_reduce
    return float(row), float(col)


def project_candidate(base, camera, prediction, foot_anchor):
    box = np.asarray(prediction["box"], dtype=np.float64)
    pose_foot = np.asarray(prediction["foot"], dtype=np.float64)
    if foot_anchor == "bbox_bottom":
        image_foot = np.asarray([(box[0] + box[2]) / 2.0, box[3]], dtype=np.float64)
    elif foot_anchor == "pose_x_bbox_y":
        image_foot = np.asarray([pose_foot[0], box[3]], dtype=np.float64)
    else:
        image_foot = pose_foot

    world_cm = get_worldcoord_from_imagecoord(
        image_foot.reshape(2, 1),
        base.intrinsic_matrices[camera],
        base.extrinsic_matrices[camera],
    )[:, 0]
    grid = base.get_worldgrid_from_worldcoord(world_cm)
    grid_x, grid_y = float(grid[0]), float(grid[1])
    valid = (
        math.isfinite(grid_x)
        and math.isfinite(grid_y)
        and 0 <= grid_x < base.worldgrid_shape[0]
        and 0 <= grid_y < base.worldgrid_shape[1]
    )
    if not valid:
        return None
    return {
        "camera": camera,
        "world_m": (world_cm / 100.0).astype(np.float64),
        "grid_x": grid_x,
        "grid_y": grid_y,
        "image_foot": image_foot.tolist(),
        "confidence": float(prediction["confidence"]),
        "box_conf": float(prediction["box_conf"]),
        "foot_conf": float(prediction["foot_conf"]),
        "foot_source": prediction["source"],
    }


def merge_candidates(candidates, radius_m, min_views):
    clusters = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        best = None
        best_distance = float("inf")
        for cluster in clusters:
            if candidate["camera"] in cluster["cameras"]:
                continue
            distance = float(np.linalg.norm(candidate["world_m"] - cluster["center_m"]))
            if distance <= radius_m and distance < best_distance:
                best, best_distance = cluster, distance
        if best is None:
            clusters.append({
                "members": [candidate],
                "cameras": {candidate["camera"]},
                "center_m": candidate["world_m"].copy(),
            })
            continue
        best["members"].append(candidate)
        best["cameras"].add(candidate["camera"])
        weights = np.asarray([member["confidence"] for member in best["members"]])
        points = np.stack([member["world_m"] for member in best["members"]])
        best["center_m"] = np.average(points, axis=0, weights=weights)

    merged = []
    for cluster in clusters:
        if len(cluster["cameras"]) < min_views:
            continue
        distances = np.asarray([
            np.linalg.norm(member["world_m"] - cluster["center_m"])
            for member in cluster["members"]
        ])
        confidence = 1.0 - np.prod([
            1.0 - np.clip(member["confidence"], 0.0, 1.0)
            for member in cluster["members"]
        ])
        confidence *= math.exp(-float(distances.mean()) / max(radius_m, 1e-6))
        merged.append({
            "center_m": cluster["center_m"],
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "num_views": len(cluster["cameras"]),
            "cameras": sorted(camera + 1 for camera in cluster["cameras"]),
            "spread_m": float(distances.mean()),
            "members": cluster["members"],
        })
    return merged


def gated_assignment(source_points, target_points, radius_m):
    """Hungarian assignment with a hard metric-distance gate."""
    if not source_points or not target_points:
        return [], set(range(len(source_points))), set(range(len(target_points)))
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    distances = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2)
    source_indices, target_indices = linear_sum_assignment(distances)
    matches = [
        (int(source_index), int(target_index))
        for source_index, target_index in zip(source_indices, target_indices)
        if distances[source_index, target_index] <= radius_m
    ]
    matched_source = {source_index for source_index, _ in matches}
    matched_target = {target_index for _, target_index in matches}
    return (
        matches,
        set(range(len(source_points))) - matched_source,
        set(range(len(target_points))) - matched_target,
    )


def select_temporal_clusters(
    clusters, min_views, previous_tracks, singleton_conf, temporal_radius_m
):
    """Keep multi-view clusters and rescue only track-supported singletons.

    Multi-view observations claim their previous tracks first. A singleton can
    therefore only continue a still-unmatched track and cannot duplicate a
    person already represented by a current multi-view cluster.
    """
    multi_view = [cluster for cluster in clusters if cluster["num_views"] >= min_views]
    singletons = [
        cluster for cluster in clusters
        if cluster["num_views"] == 1 and cluster["confidence"] >= singleton_conf
    ]
    accepted = list(multi_view)
    if not previous_tracks:
        return accepted, [cluster["center_m"] for cluster in accepted], 0

    _, unmatched_tracks, _ = gated_assignment(
        previous_tracks,
        [cluster["center_m"] for cluster in multi_view],
        temporal_radius_m,
    )
    unmatched_track_points = [previous_tracks[index] for index in sorted(unmatched_tracks)]
    singleton_matches, _, _ = gated_assignment(
        unmatched_track_points,
        [cluster["center_m"] for cluster in singletons],
        temporal_radius_m,
    )
    rescued_indices = {singleton_index for _, singleton_index in singleton_matches}
    rescued = [singletons[index] for index in sorted(rescued_indices)]
    accepted.extend(rescued)
    return accepted, [cluster["center_m"] for cluster in accepted], len(rescued)


def load_gt_cells(base, annotation_path, grid_reduce):
    if not annotation_path.exists():
        return []
    with annotation_path.open("r", encoding="utf-8") as handle:
        people = json.load(handle)
    cells = []
    for person in people:
        grid_x, grid_y = base.get_worldgrid_from_pos(int(person["positionID"]))
        row, col = grid_to_mvdet_cell(base, grid_x, grid_y, grid_reduce)
        cells.append({
            "row": row,
            "col": col,
            "grid_x": float(grid_x),
            "grid_y": float(grid_y),
            "person_id": int(person["personID"]),
        })
    return cells


def evaluate_bev_points(merged_points, gt_cells, radius_m):
    """One-to-one BEV matching on Wildtrack's 2.5 cm world grid."""
    if not merged_points or not gt_cells:
        tp = 0
        return {
            "tp": tp,
            "fp": len(merged_points),
            "fn": len(gt_cells),
            "matched_distances_m": [],
        }

    predictions_m = np.asarray(
        [[point["grid_x"], point["grid_y"]] for point in merged_points], dtype=np.float64
    ) * 0.025
    ground_truth_m = np.asarray(
        [[point["grid_x"], point["grid_y"]] for point in gt_cells], dtype=np.float64
    ) * 0.025
    distances = np.linalg.norm(
        predictions_m[:, None, :] - ground_truth_m[None, :, :], axis=2
    )
    pred_indices, gt_indices = linear_sum_assignment(distances)
    matched = [
        float(distances[pred_idx, gt_idx])
        for pred_idx, gt_idx in zip(pred_indices, gt_indices)
        if distances[pred_idx, gt_idx] <= radius_m
    ]
    tp = len(matched)
    return {
        "tp": tp,
        "fp": len(merged_points) - tp,
        "fn": len(gt_cells) - tp,
        "matched_distances_m": matched,
    }


def oracle_coverage(points, gt_cells, radius_m):
    """Count GT locations having at least one projected point inside the gate."""
    if not points or not gt_cells:
        return 0
    predictions_m = np.asarray(
        [[point["grid_x"], point["grid_y"]] for point in points], dtype=np.float64
    ) * 0.025
    ground_truth_m = np.asarray(
        [[point["grid_x"], point["grid_y"]] for point in gt_cells], dtype=np.float64
    ) * 0.025
    distances = np.linalg.norm(
        predictions_m[:, None, :] - ground_truth_m[None, :, :], axis=2
    )
    return int((distances.min(axis=0) <= radius_m).sum())


def camera_support_histogram(raw_candidates, gt_cells, radius_m):
    """Histogram of unique cameras providing a correct raw point for each GT."""
    histogram = {}
    for gt in gt_cells:
        gt_m = np.asarray([gt["grid_x"], gt["grid_y"]], dtype=np.float64) * 0.025
        cameras = {
            int(point["camera"])
            for point in raw_candidates
            if np.linalg.norm(
                np.asarray([point["grid_x"], point["grid_y"]], dtype=np.float64) * 0.025
                - gt_m
            ) <= radius_m
        }
        support = len(cameras)
        histogram[support] = histogram.get(support, 0) + 1
    return histogram


def mvdet_map_kernel(grid_reduce):
    """Reproduce frameDataset's 41x41 BEV Gaussian kernel exactly."""
    map_variance = 20 / grid_reduce
    coordinates_x, coordinates_y = np.meshgrid(np.arange(-20, 21), np.arange(-20, 21))
    positions = np.stack([coordinates_x, coordinates_y], axis=2)
    kernel = multivariate_normal.pdf(
        positions, [0, 0], np.identity(2) * map_variance
    )
    return (kernel / kernel.max()).astype(np.float32)


def build_occupancy_maps(shape, merged_points, gt_cells, grid_reduce):
    """Build the point targets and Gaussian heatmaps consumed by MVDet's loss."""
    pseudo_points = np.zeros(shape, dtype=np.float32)
    pseudo_confidence = np.zeros(shape, dtype=np.float32)
    gt_points = np.zeros(shape, dtype=np.float32)
    for point in merged_points:
        row, col = int(point["row"]), int(point["col"])
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            pseudo_points[row, col] = 1.0
            pseudo_confidence[row, col] = max(
                pseudo_confidence[row, col], point["confidence"]
            )
    for point in gt_cells:
        row, col = int(point["row"]), int(point["col"])
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            gt_points[row, col] = 1.0

    kernel = mvdet_map_kernel(grid_reduce)
    pseudo_heatmap = convolve2d(pseudo_points, kernel, mode="same", boundary="fill")
    confidence_heatmap = convolve2d(
        pseudo_confidence, kernel, mode="same", boundary="fill"
    )
    gt_heatmap = convolve2d(gt_points, kernel, mode="same", boundary="fill")
    return {
        "pseudo_map": pseudo_points,
        "pseudo_conf_map": pseudo_confidence,
        "pseudo_heatmap": pseudo_heatmap.astype(np.float32),
        "pseudo_conf_heatmap": np.clip(confidence_heatmap, 0.0, 1.0).astype(np.float32),
        "gt_map": gt_points,
        "gt_heatmap": gt_heatmap.astype(np.float32),
    }


def configure_axis(axis, shape, title):
    axis.imshow(np.zeros(shape, dtype=np.float32), cmap="gray", vmin=0, vmax=1, origin="upper")
    axis.set_xlim(-0.5, shape[1] - 0.5)
    axis.set_ylim(shape[0] - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_xlabel("MVDet BEV column")
    axis.set_ylabel("MVDet BEV row")
    axis.grid(color="white", alpha=0.12, linewidth=0.4)


def draw_frame(output, frame, shape, raw_candidates, merged_points, gt_cells, anchor):
    figure, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    configure_axis(axes[0], shape, f"Frame {frame:08d}: projected YOLO {anchor} by camera")
    configure_axis(axes[1], shape, f"Frame {frame:08d}: multi-view merged pseudo points")

    for camera in range(7):
        camera_points = [point for point in raw_candidates if point["camera"] == camera]
        if not camera_points:
            continue
        rows = [point["row"] for point in camera_points]
        cols = [point["col"] for point in camera_points]
        sizes = [25 + 90 * point["confidence"] for point in camera_points]
        axes[0].scatter(cols, rows, s=sizes, c=CAMERA_COLORS[camera], alpha=0.72,
                        edgecolors="white", linewidths=0.35, label=f"C{camera + 1}")

    if merged_points:
        rows = [point["row"] for point in merged_points]
        cols = [point["col"] for point in merged_points]
        sizes = [45 + 150 * point["confidence"] for point in merged_points]
        axes[1].scatter(cols, rows, s=sizes, c="#ff1744", edgecolors="white",
                        linewidths=0.8, label="YOLO pseudo")
        for point in merged_points:
            axes[1].annotate(
                f"{point['num_views']}v/{point['confidence']:.2f}",
                (point["col"], point["row"]),
                xytext=(4, -6), textcoords="offset points", color="white", fontsize=6,
            )

    if gt_cells:
        gt_rows = [point["row"] for point in gt_cells]
        gt_cols = [point["col"] for point in gt_cells]
        for axis in axes:
            axis.scatter(gt_cols, gt_rows, marker="x", s=34, c="#00e676",
                         linewidths=1.2, label="full GT")

    if raw_candidates:
        axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    if merged_points or gt_cells:
        axes[1].legend(loc="upper right", fontsize=8)
    figure.suptitle(
        f"Wildtrack/MVDet grid {shape[0]}x{shape[1]} | raw={len(raw_candidates)} | "
        f"merged={len(merged_points)}",
        fontsize=12,
    )
    destination = output / f"{frame:08d}_bev_projection.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def draw_occupancy_frame(output, frame, occupancy_maps, merged_points, gt_cells, evaluation):
    """Render YOLO's final confidence-weighted occupancy map beside full GT."""
    figure, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    pseudo_image = axes[0].imshow(
        occupancy_maps["pseudo_conf_heatmap"], cmap="magma", vmin=0, vmax=1,
        origin="upper",
    )
    axes[0].set_title("YOLO multi-view pedestrian occupancy map")
    figure.colorbar(pseudo_image, ax=axes[0], fraction=0.025, pad=0.02)

    gt_image = axes[1].imshow(
        np.clip(occupancy_maps["gt_heatmap"], 0, 1), cmap="viridis", vmin=0, vmax=1,
        origin="upper",
    )
    axes[1].set_title("Full GT occupancy map (evaluation only)")
    figure.colorbar(gt_image, ax=axes[1], fraction=0.025, pad=0.02)

    if merged_points:
        axes[0].scatter(
            [point["col"] for point in merged_points],
            [point["row"] for point in merged_points],
            facecolors="none", edgecolors="cyan", s=36, linewidths=0.7,
            label="merged centers",
        )
        axes[0].legend(loc="upper right", fontsize=8)
    if gt_cells:
        axes[1].scatter(
            [point["col"] for point in gt_cells],
            [point["row"] for point in gt_cells],
            marker="x", c="white", s=25, linewidths=0.8, label="GT centers",
        )
        axes[1].legend(loc="upper right", fontsize=8)

    for axis in axes:
        axis.set_xlim(-0.5, occupancy_maps["pseudo_map"].shape[1] - 0.5)
        axis.set_ylim(occupancy_maps["pseudo_map"].shape[0] - 0.5, -0.5)
        axis.set_aspect("equal")
        axis.set_xlabel("MVDet BEV column")
        axis.set_ylabel("MVDet BEV row")

    metrics = ""
    if evaluation is not None:
        metrics = (
            f" | TP={evaluation['tp']} FP={evaluation['fp']} FN={evaluation['fn']}"
        )
    figure.suptitle(
        f"Frame {frame:08d} | output shape 120x360 | people={len(merged_points)}{metrics}",
        fontsize=12,
    )
    destination = output / f"{frame:08d}_occupancy_map.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def main():
    args = parse_args()
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.grid_reduce <= 0:
        raise ValueError("--grid-reduce must be positive")
    if args.eval_radius_m <= 0:
        raise ValueError("--eval-radius-m must be positive")
    if args.temporal_radius_m <= 0:
        raise ValueError("--temporal-radius-m must be positive")
    if not 0 <= args.temporal_conf <= 1:
        raise ValueError("--temporal-conf must be in [0, 1]")

    image_subsets = args.image_subsets.resolve()
    data_root = image_subsets.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output))

    base = Wildtrack(str(data_root))
    image_paths = base.get_image_fpaths(range(args.start_frame, base.num_frame))
    frame_sets = [set(image_paths[camera]) for camera in range(base.num_cam)]
    frames = sorted(set.intersection(*frame_sets))[: args.num_frames]
    if not frames:
        raise RuntimeError("No synchronized C1-C7 frames found")

    from ultralytics import YOLO

    model = YOLO(args.model)
    reduced_shape = tuple(int(value / args.grid_reduce) for value in base.worldgrid_shape)
    frame_summaries = []
    rendered = []
    evaluation_totals = {"tp": 0, "fp": 0, "fn": 0, "matched_distances_m": []}
    fusion_totals = {
        "gt": 0,
        "raw_points": 0,
        "final_points": 0,
        "raw_oracle_covered": 0,
        "final_oracle_covered": 0,
        "all_clusters": 0,
        "singleton_clusters": 0,
        "temporal_singletons_rescued": 0,
        "camera_support_histogram": {},
    }
    previous_tracks = []

    for frame_index, frame in enumerate(frames, start=1):
        raw_candidates = []
        camera_counts = {}
        for start_camera in range(0, base.num_cam, args.inference_batch):
            cameras = list(range(start_camera, min(start_camera + args.inference_batch, base.num_cam)))
            results = model.predict(
                source=[image_paths[camera][frame] for camera in cameras],
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
                save=False,
            )
            for camera, result in zip(cameras, results):
                accepted = 0
                for prediction in extract_predictions(result, args.kpt_conf):
                    if prediction["confidence"] < args.candidate_conf:
                        continue
                    candidate = project_candidate(base, camera, prediction, args.foot_anchor)
                    if candidate is None:
                        continue
                    row, col = grid_to_mvdet_cell(
                        base, candidate["grid_x"], candidate["grid_y"], args.grid_reduce
                    )
                    candidate["row"], candidate["col"] = row, col
                    raw_candidates.append(candidate)
                    accepted += 1
                camera_counts[f"C{camera + 1}"] = accepted

        gt_cells = load_gt_cells(
            base, data_root / "annotations_positions" / f"{frame:08d}.json", args.grid_reduce
        ) if args.show_gt else []
        all_clusters = merge_candidates(raw_candidates, args.merge_radius_m, min_views=1)
        if args.temporal_singletons:
            clusters, previous_tracks, rescued_singletons = select_temporal_clusters(
                all_clusters,
                args.min_views,
                previous_tracks,
                args.temporal_conf,
                args.temporal_radius_m,
            )
        else:
            clusters = [
                cluster for cluster in all_clusters
                if cluster["num_views"] >= args.min_views
            ]
            previous_tracks = [cluster["center_m"] for cluster in clusters]
            rescued_singletons = 0
        merged_points = []
        raw_count_map = np.zeros(reduced_shape, dtype=np.int16)
        for candidate in raw_candidates:
            row, col = int(candidate["row"]), int(candidate["col"])
            if 0 <= row < reduced_shape[0] and 0 <= col < reduced_shape[1]:
                raw_count_map[row, col] += 1
        for cluster in clusters:
            world_cm = cluster["center_m"] * 100.0
            grid_x, grid_y = base.get_worldgrid_from_worldcoord(world_cm)
            row, col = grid_to_mvdet_cell(base, grid_x, grid_y, args.grid_reduce)
            point = {
                "row": row,
                "col": col,
                "grid_x": float(grid_x),
                "grid_y": float(grid_y),
                "confidence": cluster["confidence"],
                "num_views": cluster["num_views"],
                "cameras": cluster["cameras"],
                "spread_m": cluster["spread_m"],
            }
            merged_points.append(point)
        frame_evaluation = (
            evaluate_bev_points(merged_points, gt_cells, args.eval_radius_m)
            if args.show_gt else None
        )
        if frame_evaluation is not None:
            for key in ("tp", "fp", "fn"):
                evaluation_totals[key] += frame_evaluation[key]
            evaluation_totals["matched_distances_m"].extend(
                frame_evaluation["matched_distances_m"]
            )
        raw_oracle_covered = (
            oracle_coverage(raw_candidates, gt_cells, args.eval_radius_m)
            if args.show_gt else None
        )
        final_oracle_covered = (
            oracle_coverage(merged_points, gt_cells, args.eval_radius_m)
            if args.show_gt else None
        )
        support_histogram = (
            camera_support_histogram(raw_candidates, gt_cells, args.eval_radius_m)
            if args.show_gt else {}
        )
        fusion_totals["gt"] += len(gt_cells)
        fusion_totals["raw_points"] += len(raw_candidates)
        fusion_totals["final_points"] += len(merged_points)
        fusion_totals["raw_oracle_covered"] += raw_oracle_covered or 0
        fusion_totals["final_oracle_covered"] += final_oracle_covered or 0
        fusion_totals["all_clusters"] += len(all_clusters)
        fusion_totals["singleton_clusters"] += sum(
            cluster["num_views"] == 1 for cluster in all_clusters
        )
        fusion_totals["temporal_singletons_rescued"] += rescued_singletons
        for support, count in support_histogram.items():
            key = str(support)
            fusion_totals["camera_support_histogram"][key] = (
                fusion_totals["camera_support_histogram"].get(key, 0) + count
            )
        rendered.append(draw_frame(
            output, frame, reduced_shape, raw_candidates, merged_points, gt_cells,
            args.foot_anchor,
        ))
        occupancy_maps = build_occupancy_maps(
            reduced_shape, merged_points, gt_cells, args.grid_reduce
        )
        occupancy_render = draw_occupancy_frame(
            output, frame, occupancy_maps, merged_points, gt_cells, frame_evaluation
        )
        np.savez_compressed(
            output / f"{frame:08d}_mvdet_maps.npz",
            raw_count_map=raw_count_map,
            **occupancy_maps,
        )

        serializable_candidates = [
            {key: value.tolist() if isinstance(value, np.ndarray) else value
             for key, value in candidate.items()}
            for candidate in raw_candidates
        ]
        frame_summaries.append({
            "frame": frame,
            "camera_counts": camera_counts,
            "raw_candidates": serializable_candidates,
            "merged_points": merged_points,
            "gt_count": len(gt_cells) if args.show_gt else None,
            "evaluation": frame_evaluation,
            "raw_oracle_covered": raw_oracle_covered,
            "raw_oracle_recall": (
                raw_oracle_covered / len(gt_cells) if gt_cells else None
            ),
            "final_oracle_covered": final_oracle_covered,
            "camera_support_histogram": {
                str(key): value for key, value in sorted(support_histogram.items())
            },
            "all_clusters": len(all_clusters),
            "singleton_clusters": sum(
                cluster["num_views"] == 1 for cluster in all_clusters
            ),
            "temporal_singletons_rescued": rescued_singletons,
            "occupancy_map": occupancy_render.name,
        })
        print(
            f"[{frame_index}/{len(frames)}] frame={frame:08d} "
            f"raw={len(raw_candidates)} merged={len(merged_points)} output={rendered[-1].name}"
        )

    summary = {
        "data_root": str(data_root),
        "image_subsets": str(image_subsets),
        "model": args.model,
        "frames": frames,
        "grid_reduce": args.grid_reduce,
        "mvdet_bev_shape": list(reduced_shape),
        "foot_anchor": args.foot_anchor,
        "min_views": args.min_views,
        "merge_radius_m": args.merge_radius_m,
        "temporal_singletons": args.temporal_singletons,
        "temporal_conf": args.temporal_conf,
        "temporal_radius_m": args.temporal_radius_m,
        "frame_results": frame_summaries,
    }
    if args.show_gt:
        tp = evaluation_totals["tp"]
        fp = evaluation_totals["fp"]
        fn = evaluation_totals["fn"]
        distances = evaluation_totals.pop("matched_distances_m")
        summary["gt_evaluation"] = {
            "match_radius_m": args.eval_radius_m,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "moda": 1.0 - (fp + fn) / max(tp + fn, 1),
            "mean_error_m": float(np.mean(distances)) if distances else None,
            "median_error_m": float(np.median(distances)) if distances else None,
        }
        total_gt = fusion_totals["gt"]
        fusion_totals["raw_oracle_recall"] = (
            fusion_totals["raw_oracle_covered"] / max(total_gt, 1)
        )
        fusion_totals["final_oracle_recall"] = (
            fusion_totals["final_oracle_covered"] / max(total_gt, 1)
        )
        fusion_totals["premerge_misses"] = (
            total_gt - fusion_totals["raw_oracle_covered"]
        )
        fusion_totals["fusion_coverage_losses"] = (
            fusion_totals["raw_oracle_covered"]
            - fusion_totals["final_oracle_covered"]
        )
        summary["fusion_diagnostics"] = fusion_totals
    with (output / "projection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "frame_results"}, indent=2))


if __name__ == "__main__":
    main()
