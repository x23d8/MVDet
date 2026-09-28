"""Generate confidence-weighted BEV pseudo labels from multi-view YOLO pose feet.

Run this offline before MVDet training.  The training loader consumes the JSON
files written here; YOLO is deliberately not executed inside each epoch.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TOOLS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from multiview_detector.datasets.MultiviewX import MultiviewX
from multiview_detector.datasets.Wildtrack import Wildtrack
from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
from yolo_pose_foot_demo import extract_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLO-pose BEV pseudo labels")
    parser.add_argument("--dataset", choices=["wildtrack", "multiviewx"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo26m-pose.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--kpt-conf", type=float, default=0.35)
    parser.add_argument("--candidate-conf", type=float, default=0.15)
    parser.add_argument(
        "--foot-anchor",
        choices=["pose", "bbox_bottom", "pose_x_bbox_y"],
        default="pose_x_bbox_y",
        help="image point projected to the ground plane",
    )
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--merge-radius-m", type=float, default=0.75)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--inference-batch", type=int, default=4,
                        help="number of camera images inferred together")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate-gt", action="store_true",
                        help="report demo-only BEV precision/recall against full annotations")
    parser.add_argument("--eval-radius-m", type=float, default=0.5)
    return parser.parse_args()


def world_units_to_meters(dataset_name, world_coord):
    # Wildtrack calibration is in centimetres; MultiviewX is in metres.
    return world_coord / 100.0 if dataset_name == "wildtrack" else world_coord


def meters_to_world_units(dataset_name, world_coord_m):
    return world_coord_m * 100.0 if dataset_name == "wildtrack" else world_coord_m


def project_candidate(base, dataset_name, camera_index, prediction, foot_anchor):
    box = np.asarray(prediction["box"], dtype=np.float64)
    pose_foot = np.asarray(prediction["foot"], dtype=np.float64)
    if foot_anchor == "bbox_bottom":
        selected_foot = np.asarray([(box[0] + box[2]) / 2.0, box[3]], dtype=np.float64)
    elif foot_anchor == "pose_x_bbox_y":
        selected_foot = np.asarray([pose_foot[0], box[3]], dtype=np.float64)
    else:
        selected_foot = pose_foot
    image_coord = selected_foot.reshape(2, 1)
    world_coord = get_worldcoord_from_imagecoord(
        image_coord,
        base.intrinsic_matrices[camera_index],
        base.extrinsic_matrices[camera_index],
    )[:, 0]
    world_coord_m = world_units_to_meters(dataset_name, world_coord)
    world_grid = base.get_worldgrid_from_worldcoord(world_coord)
    grid_x, grid_y = float(world_grid[0]), float(world_grid[1])
    if not (math.isfinite(grid_x) and math.isfinite(grid_y)):
        return None
    if base.indexing == "xy":
        valid = 0 <= grid_x < base.worldgrid_shape[1] and 0 <= grid_y < base.worldgrid_shape[0]
    else:
        valid = 0 <= grid_x < base.worldgrid_shape[0] and 0 <= grid_y < base.worldgrid_shape[1]
    if not valid:
        return None
    return {
        "camera": camera_index,
        "world_m": world_coord_m.astype(np.float64),
        "grid_x": grid_x,
        "grid_y": grid_y,
        "confidence": float(prediction["confidence"]),
        "box_conf": float(prediction["box_conf"]),
        "foot_conf": float(prediction["foot_conf"]),
        "foot_source": f"{prediction['source']}:{foot_anchor}",
        "image_foot": [float(value) for value in selected_foot],
    }


def merge_candidates(candidates, radius_m, min_views):
    """Greedy confidence-first clustering with at most one point per camera."""
    clusters = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        best_cluster = None
        best_distance = float("inf")
        for cluster in clusters:
            if candidate["camera"] in cluster["cameras"]:
                continue
            distance = float(np.linalg.norm(candidate["world_m"] - cluster["center_m"]))
            if distance <= radius_m and distance < best_distance:
                best_cluster = cluster
                best_distance = distance
        if best_cluster is None:
            clusters.append(
                {
                    "members": [candidate],
                    "cameras": {candidate["camera"]},
                    "center_m": candidate["world_m"].copy(),
                }
            )
            continue
        best_cluster["members"].append(candidate)
        best_cluster["cameras"].add(candidate["camera"])
        weights = np.asarray([member["confidence"] for member in best_cluster["members"]])
        points = np.stack([member["world_m"] for member in best_cluster["members"]])
        best_cluster["center_m"] = np.average(points, axis=0, weights=weights)

    merged = []
    for cluster in clusters:
        if len(cluster["cameras"]) < min_views:
            continue
        members = cluster["members"]
        distances = np.asarray(
            [np.linalg.norm(member["world_m"] - cluster["center_m"]) for member in members],
            dtype=np.float64,
        )
        independent_confidence = 1.0 - np.prod(
            [1.0 - np.clip(member["confidence"], 0.0, 1.0) for member in members]
        )
        consistency = math.exp(-float(distances.mean()) / max(radius_m, 1e-6))
        merged.append(
            {
                "center_m": cluster["center_m"],
                "confidence": float(np.clip(independent_confidence * consistency, 0.0, 1.0)),
                "num_views": len(cluster["cameras"]),
                "spread_m": float(distances.mean()),
                "members": members,
            }
        )
    return merged


def evaluate_generated_labels(base, dataset_name, data_root, output, frames, radius_m):
    tp = fp = fn = 0
    matched_distances = []
    annotation_dir = data_root / "annotations_positions"
    for frame in frames:
        pseudo_path = output / f"{frame:08d}.json"
        annotation_path = annotation_dir / f"{frame:08d}.json"
        if not pseudo_path.exists() or not annotation_path.exists():
            continue
        with pseudo_path.open("r", encoding="utf-8") as handle:
            pseudo_payload = json.load(handle)
        with annotation_path.open("r", encoding="utf-8") as handle:
            annotations = json.load(handle)
        predicted = []
        for point in pseudo_payload.get("points", []):
            world = base.get_worldcoord_from_worldgrid(
                np.asarray([point["grid_x"], point["grid_y"]], dtype=np.float64)
            )
            predicted.append(world_units_to_meters(dataset_name, world))
        ground_truth = []
        for person in annotations:
            if not any(
                view["xmin"] >= 0 and view["ymin"] >= 0 and view["xmax"] >= 0 and view["ymax"] >= 0
                for view in person["views"]
            ):
                continue
            world = base.get_worldcoord_from_pos(int(person["positionID"]))
            ground_truth.append(world_units_to_meters(dataset_name, world))
        if not predicted or not ground_truth:
            tp += 0
            fp += len(predicted)
            fn += len(ground_truth)
            continue
        distance_matrix = np.linalg.norm(
            np.stack(predicted)[:, None, :] - np.stack(ground_truth)[None, :, :], axis=2
        )
        pred_indices, gt_indices = linear_sum_assignment(distance_matrix)
        valid_distances = [
            float(distance_matrix[pred_idx, gt_idx])
            for pred_idx, gt_idx in zip(pred_indices, gt_indices)
            if distance_matrix[pred_idx, gt_idx] <= radius_m
        ]
        frame_tp = len(valid_distances)
        tp += frame_tp
        fp += len(predicted) - frame_tp
        fn += len(ground_truth) - frame_tp
        matched_distances.extend(valid_distances)
    return {
        "match_radius_m": radius_m,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "mean_localization_error_m": float(np.mean(matched_distances)) if matched_distances else None,
        "median_localization_error_m": float(np.median(matched_distances)) if matched_distances else None,
    }


def main():
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_DIR / "Ultralytics").resolve()))
    from ultralytics import YOLO

    data_root = args.data_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = Wildtrack(str(data_root)) if args.dataset == "wildtrack" else MultiviewX(str(data_root))
    default_end = int(base.num_frame * args.train_ratio)
    end_frame = args.end_frame if args.end_frame is not None else default_end
    image_paths = base.get_image_fpaths(range(args.start_frame, end_frame))
    frame_sets = [set(image_paths[camera].keys()) for camera in range(base.num_cam)]
    frames = sorted(set.intersection(*frame_sets))
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise RuntimeError("No synchronized frames found for the requested range")

    model = YOLO(args.model)
    totals = {"frames": 0, "camera_candidates": 0, "bev_pseudo_points": 0}
    for frame_index, frame in enumerate(frames, start=1):
        destination = output / f"{frame:08d}.json"
        if destination.exists() and not args.overwrite:
            with destination.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            totals["frames"] += 1
            totals["camera_candidates"] += sum(
                existing.get("camera_candidate_counts", {}).values()
            )
            totals["bev_pseudo_points"] += len(existing.get("points", []))
            continue
        projected = []
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
                predictions = extract_predictions(result, args.kpt_conf)
                accepted = 0
                for prediction in predictions:
                    if prediction["confidence"] < args.candidate_conf:
                        continue
                    candidate = project_candidate(
                        base, args.dataset, camera, prediction, args.foot_anchor
                    )
                    if candidate is not None:
                        projected.append(candidate)
                        accepted += 1
                camera_counts[str(camera + 1)] = accepted

        merged = merge_candidates(projected, args.merge_radius_m, args.min_views)
        points = []
        for cluster in merged:
            world_coord = meters_to_world_units(args.dataset, cluster["center_m"])
            grid = base.get_worldgrid_from_worldcoord(world_coord)
            points.append(
                {
                    "grid_x": float(grid[0]),
                    "grid_y": float(grid[1]),
                    "confidence": cluster["confidence"],
                    "num_views": cluster["num_views"],
                    "spread_m": cluster["spread_m"],
                    "members": [
                        {
                            key: value
                            for key, value in member.items()
                            if key not in {"world_m"}
                        }
                        for member in cluster["members"]
                    ],
                }
            )
        payload = {
            "frame": frame,
            "model": args.model,
            "settings": {
                "imgsz": args.imgsz,
                "det_conf": args.conf,
                "keypoint_conf": args.kpt_conf,
                "candidate_conf": args.candidate_conf,
                "foot_anchor": args.foot_anchor,
                "min_views": args.min_views,
                "merge_radius_m": args.merge_radius_m,
            },
            "camera_candidate_counts": camera_counts,
            "points": points,
        }
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        totals["frames"] += 1
        totals["camera_candidates"] += len(projected)
        totals["bev_pseudo_points"] += len(points)
        print(
            f"[{frame_index}/{len(frames)}] frame={frame:08d} "
            f"candidates={len(projected)} pseudo={len(points)}"
        )

    summary = {
        "dataset": args.dataset,
        "data_root": str(data_root),
        "output": str(output),
        "model": args.model,
        **totals,
    }
    if args.evaluate_gt:
        summary["gt_evaluation"] = evaluate_generated_labels(
            base, args.dataset, data_root, output, frames, args.eval_radius_m
        )
    with (output / "generation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
