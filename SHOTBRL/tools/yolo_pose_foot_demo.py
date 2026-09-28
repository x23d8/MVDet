"""Run a small YOLO footprint demo and compare it with Wildtrack boxes.

The script intentionally keeps model inference outside the MVDet training loop.  It
produces annotated images plus JSON/CSV summaries that can be inspected before
generating BEV pseudo labels for the full training set.  By default it mirrors the
tracktacular-pseudoloss branch: YOLO26s detections and the bottom-center of each
person bounding box as the image-plane footprint.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO footprint detection on one or more Wildtrack cameras.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--dataset-root",
        type=Path,
        help="Wildtrack root containing Image_subsets and annotations_positions.",
    )
    inputs.add_argument(
        "--source",
        type=Path,
        help="Backward-compatible input: a C1..C7 folder, Image_subsets, or Wildtrack root.",
    )
    parser.add_argument("--output", type=Path, default=Path("output/yolo_foot_demo"))
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["all"],
        metavar="CAMERA",
        help="Cameras to process, for example: all, C1 C2 C7, or 1 2 7.",
    )
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Frames to infer per camera; 0 processes every frame.",
    )
    parser.add_argument(
        "--save-frames",
        type=int,
        default=-1,
        help="Annotated images saved per camera; -1 saves all processed frames, 0 saves none.",
    )
    parser.add_argument(
        "--save-stride",
        type=int,
        default=1,
        help="Only consider every Nth processed frame when saving annotated images.",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--kpt-conf", type=float, default=0.35)
    parser.add_argument(
        "--foot-mode",
        choices=("bbox_bottom", "pose", "auto"),
        default="bbox_bottom",
        help=(
            "bbox_bottom matches tracktacular-pseudoloss; pose uses ankle keypoints; "
            "auto uses pose when available and bbox bottom otherwise."
        ),
    )
    parser.add_argument("--iou-match", type=float, default=0.30)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--clear-cuda-every",
        type=int,
        default=0,
        metavar="N",
        help="Call torch.cuda.empty_cache() every N frames; 0 disables it.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Override the zero-based camera index for a single camera source.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Wildtrack annotations_positions. Defaults to <dataset root>/annotations_positions.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show Ultralytics inference logs.")
    parser.add_argument("--no-json", action="store_true", help="Do not write summary.json.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write predictions.csv.")
    return parser.parse_args()


def bbox_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-9)


def load_gt(annotation_path, camera_index):
    if not annotation_path.exists():
        return []
    with annotation_path.open("r", encoding="utf-8") as handle:
        people = json.load(handle)
    boxes = []
    for person in people:
        view = person["views"][camera_index]
        if min(view["xmin"], view["ymin"], view["xmax"], view["ymax"]) < 0:
            continue
        box = np.asarray(
            [view["xmin"], view["ymin"], view["xmax"], view["ymax"]],
            dtype=np.float32,
        )
        boxes.append(
            {
                "person_id": int(person["personID"]),
                "box": box,
                "foot": np.asarray([(box[0] + box[2]) / 2.0, box[3]], dtype=np.float32),
            }
        )
    return boxes


def extract_predictions(result, kpt_conf, foot_mode="auto"):
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    box_scores = result.boxes.conf.detach().cpu().numpy()
    keypoints = None if result.keypoints is None else result.keypoints.data.detach().cpu().numpy()
    predictions = []
    for index, (box, box_score) in enumerate(zip(boxes, box_scores)):
        bbox_bottom = np.asarray([(box[0] + box[2]) / 2.0, box[3]], dtype=np.float32)
        if foot_mode == "bbox_bottom" or keypoints is None:
            if foot_mode == "pose" and keypoints is None:
                continue
            predictions.append(
                {
                    "box": box.astype(np.float32),
                    "box_conf": float(box_score),
                    "foot": bbox_bottom,
                    "foot_conf": 1.0,
                    "confidence": float(box_score),
                    "source": "bbox_bottom",
                    "left_ankle": None,
                    "right_ankle": None,
                }
            )
            continue

        pose = keypoints[index]
        left, right = pose[LEFT_ANKLE], pose[RIGHT_ANKLE]
        valid_left = left.shape[0] >= 3 and float(left[2]) >= kpt_conf
        valid_right = right.shape[0] >= 3 and float(right[2]) >= kpt_conf
        if valid_left and valid_right:
            foot = (left[:2] + right[:2]) / 2.0
            foot_conf = float((left[2] + right[2]) / 2.0)
            source = "ankle_midpoint"
        elif valid_left or valid_right:
            ankle = left if valid_left else right
            foot = 0.5 * ankle[:2] + 0.5 * bbox_bottom
            foot_conf = float(ankle[2]) * 0.75
            source = "single_ankle_bbox"
        else:
            foot = bbox_bottom
            foot_conf = 0.50
            source = "bbox_bottom"
        predictions.append(
            {
                "box": box.astype(np.float32),
                "box_conf": float(box_score),
                "foot": foot.astype(np.float32),
                "foot_conf": foot_conf,
                "confidence": float(box_score) * foot_conf,
                "source": source,
                "left_ankle": left.tolist(),
                "right_ankle": right.tolist(),
            }
        )
    return predictions


def match_predictions(predictions, ground_truth, iou_threshold):
    if not predictions or not ground_truth:
        return [], list(range(len(predictions))), list(range(len(ground_truth)))
    ious = np.zeros((len(predictions), len(ground_truth)), dtype=np.float32)
    for pred_idx, prediction in enumerate(predictions):
        for gt_idx, gt in enumerate(ground_truth):
            ious[pred_idx, gt_idx] = bbox_iou(prediction["box"], gt["box"])
    row_indices, column_indices = linear_sum_assignment(1.0 - ious)
    matches = []
    matched_pred = set()
    matched_gt = set()
    for pred_idx, gt_idx in zip(row_indices.tolist(), column_indices.tolist()):
        if float(ious[pred_idx, gt_idx]) < iou_threshold:
            continue
        foot_error = float(np.linalg.norm(predictions[pred_idx]["foot"] - ground_truth[gt_idx]["foot"]))
        matches.append((pred_idx, gt_idx, float(ious[pred_idx, gt_idx]), foot_error))
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
    unmatched_pred = [idx for idx in range(len(predictions)) if idx not in matched_pred]
    unmatched_gt = [idx for idx in range(len(ground_truth)) if idx not in matched_gt]
    return matches, unmatched_pred, unmatched_gt


def draw_frame(image, predictions, ground_truth, matches, unmatched_pred, unmatched_gt):
    match_by_pred = {pred_idx: (gt_idx, iou, error) for pred_idx, gt_idx, iou, error in matches}
    for gt_idx, gt in enumerate(ground_truth):
        color = (0, 180, 0) if gt_idx not in unmatched_gt else (0, 0, 255)
        box = gt["box"].astype(int)
        foot = gt["foot"].astype(int)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.circle(image, tuple(foot), 5, (0, 255, 0), -1)
        cv2.putText(image, f"GT {gt['person_id']}", (box[0], max(15, box[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    for pred_idx, prediction in enumerate(predictions):
        box = prediction["box"].astype(int)
        foot = prediction["foot"].astype(int)
        matched = pred_idx in match_by_pred
        color = (255, 180, 0) if matched else (255, 0, 255)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.circle(image, tuple(foot), 6, (0, 255, 255), -1)
        label = f"YOLO {prediction['confidence']:.2f} {prediction['source']}"
        if matched:
            gt_idx, iou, error = match_by_pred[pred_idx]
            gt_foot = ground_truth[gt_idx]["foot"].astype(int)
            cv2.line(image, tuple(foot), tuple(gt_foot), (255, 255, 255), 1)
            label += f" IoU={iou:.2f} foot={error:.1f}px"
        cv2.putText(image, label, (box[0], min(image.shape[0] - 8, box[3] + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    cv2.putText(
        image,
        "GT=green/red(unmatched), YOLO=cyan box + yellow foot, white line=foot error",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return image


def normalize_camera_name(value):
    value = str(value).strip().upper()
    if value.isdigit():
        value = f"C{value}"
    if not (value.startswith("C") and value[1:].isdigit() and 1 <= int(value[1:]) <= 7):
        raise ValueError(f"Invalid camera {value!r}; expected C1..C7, 1..7, or all")
    return value


def resolve_camera_inputs(args):
    requested_path = (args.dataset_root or args.source).resolve()
    if not requested_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {requested_path}")

    direct_camera = requested_path.name.upper().startswith("C") and requested_path.name[1:].isdigit()
    if direct_camera:
        camera_dirs = [requested_path]
        dataset_root = requested_path.parent.parent if requested_path.parent.name == "Image_subsets" else requested_path.parent
    elif (requested_path / "Image_subsets").is_dir():
        dataset_root = requested_path
        image_root = requested_path / "Image_subsets"
        camera_dirs = [path for path in image_root.iterdir() if path.is_dir()]
    elif requested_path.name == "Image_subsets":
        dataset_root = requested_path.parent
        camera_dirs = [path for path in requested_path.iterdir() if path.is_dir()]
    else:
        raise FileNotFoundError(
            f"{requested_path} is not a Wildtrack root, Image_subsets directory, or camera directory"
        )

    available = {}
    for camera_dir in camera_dirs:
        try:
            camera_name = normalize_camera_name(camera_dir.name)
        except ValueError:
            continue
        available[camera_name] = camera_dir

    requested_cameras = [str(value).lower() for value in args.cameras]
    if "all" in requested_cameras:
        if len(requested_cameras) != 1:
            raise ValueError("Use --cameras all by itself, or list individual cameras")
        selected_names = sorted(available, key=lambda name: int(name[1:]))
    else:
        selected_names = [normalize_camera_name(value) for value in args.cameras]

    missing = [name for name in selected_names if name not in available]
    if missing:
        raise FileNotFoundError(f"Camera directories not found: {', '.join(missing)}")
    if not selected_names:
        raise FileNotFoundError(f"No C1..C7 camera directories found below {requested_path}")

    annotations = args.annotations.resolve() if args.annotations else dataset_root / "annotations_positions"
    if not annotations.is_dir():
        raise FileNotFoundError(f"annotations_positions directory not found: {annotations}")
    return [(name, available[name]) for name in selected_names], annotations


def write_reports(output, summary, csv_rows, args):
    output.mkdir(parents=True, exist_ok=True)
    if not args.no_json:
        with (output / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    if not args.no_csv:
        fieldnames = [
            "camera",
            "frame",
            "box",
            "foot",
            "box_conf",
            "foot_conf",
            "confidence",
            "source",
            "matched_gt",
            "iou",
            "foot_error_px",
        ]
        with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)


def process_camera(model, camera_name, source, output, annotations, camera_index, args):
    image_paths = sorted(path for path in source.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.num_frames > 0:
        image_paths = image_paths[: args.num_frames]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source}")

    output.mkdir(parents=True, exist_ok=True)
    print(f"[{camera_name}] Processing {len(image_paths)} frame(s) from {source}")
    frame_summaries = []
    csv_rows = []
    total_tp = total_fp = total_fn = 0
    all_foot_errors = []
    saved_images = 0
    for frame_offset, image_path in enumerate(image_paths):
        # Run exactly one image at a time. Passing the complete path list to
        # Ultralytics can create a very large batch before streaming results.
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            classes=[0],
            device=args.device,
            verbose=args.verbose,
            save=False,
        )[0]
        frame = int(image_path.stem)
        predictions = extract_predictions(result, args.kpt_conf, args.foot_mode)
        ground_truth = load_gt(annotations / f"{image_path.stem}.json", camera_index)
        matches, unmatched_pred, unmatched_gt = match_predictions(predictions, ground_truth, args.iou_match)
        total_tp += len(matches)
        total_fp += len(unmatched_pred)
        total_fn += len(unmatched_gt)
        all_foot_errors.extend(match[3] for match in matches)

        annotated_path = None
        within_save_limit = args.save_frames < 0 or saved_images < args.save_frames
        if within_save_limit and frame_offset % args.save_stride == 0:
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            annotated = draw_frame(image, predictions, ground_truth, matches, unmatched_pred, unmatched_gt)
            annotated_path = output / f"{image_path.stem}_{args.foot_mode}_foot.jpg"
            if not cv2.imwrite(str(annotated_path), annotated):
                raise RuntimeError(f"Could not write image: {annotated_path}")
            saved_images += 1

        frame_summary = {
            "camera": camera_name,
            "frame": frame,
            "image": str(image_path),
            "annotated_image": None if annotated_path is None else str(annotated_path),
            "num_gt": len(ground_truth),
            "num_predictions": len(predictions),
            "tp": len(matches),
            "fp": len(unmatched_pred),
            "fn": len(unmatched_gt),
            "mean_foot_error_px": float(np.mean([match[3] for match in matches])) if matches else None,
            "predictions": [],
        }
        match_by_pred = {match[0]: match for match in matches}
        for pred_idx, prediction in enumerate(predictions):
            match = match_by_pred.get(pred_idx)
            record = {
                "box": prediction["box"].tolist(),
                "foot": prediction["foot"].tolist(),
                "box_conf": prediction["box_conf"],
                "foot_conf": prediction["foot_conf"],
                "confidence": prediction["confidence"],
                "source": prediction["source"],
                "matched_gt": None if match is None else int(ground_truth[match[1]]["person_id"]),
                "iou": None if match is None else match[2],
                "foot_error_px": None if match is None else match[3],
            }
            frame_summary["predictions"].append(record)
            csv_rows.append({"camera": camera_name, "frame": frame, **record})
        frame_summaries.append(frame_summary)

        del result
        if args.clear_cuda_every > 0 and (frame_offset + 1) % args.clear_cuda_every == 0:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    summary = {
        "camera": camera_name,
        "camera_index": camera_index,
        "model": args.model,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "foot_mode": args.foot_mode,
        "keypoint_conf": args.kpt_conf,
        "iou_match": args.iou_match,
        "frames": len(image_paths),
        "saved_images": saved_images,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "mean_foot_error_px": float(np.mean(all_foot_errors)) if all_foot_errors else None,
        "median_foot_error_px": float(np.median(all_foot_errors)) if all_foot_errors else None,
        "per_frame": frame_summaries,
    }
    write_reports(output, summary, csv_rows, args)
    compact = {key: value for key, value in summary.items() if key != "per_frame"}
    print(json.dumps(compact, indent=2))
    return summary, csv_rows, all_foot_errors


def main():
    args = parse_args()
    if args.num_frames < 0:
        raise ValueError("--num-frames must be 0 (all frames) or a positive integer")
    if args.save_frames < -1:
        raise ValueError("--save-frames must be -1 (all), 0 (none), or a positive integer")
    if args.save_stride < 1:
        raise ValueError("--save-stride must be at least 1")
    if args.clear_cuda_every < 0:
        raise ValueError("--clear-cuda-every must be 0 or a positive integer")

    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / "Ultralytics").resolve()))
    from ultralytics import YOLO

    camera_inputs, annotations = resolve_camera_inputs(args)
    if args.camera_index is not None and len(camera_inputs) != 1:
        raise ValueError("--camera-index can only override a single selected camera")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    multi_camera = len(camera_inputs) > 1
    camera_summaries = []
    all_csv_rows = []
    all_foot_errors = []

    for camera_name, source in camera_inputs:
        camera_index = args.camera_index if args.camera_index is not None else int(camera_name[1:]) - 1
        camera_output = output / camera_name if multi_camera else output
        summary, csv_rows, foot_errors = process_camera(
            model,
            camera_name,
            source,
            camera_output,
            annotations,
            camera_index,
            args,
        )
        camera_summaries.append(summary)
        all_csv_rows.extend(csv_rows)
        all_foot_errors.extend(foot_errors)

    if multi_camera:
        total_tp = sum(summary["tp"] for summary in camera_summaries)
        total_fp = sum(summary["fp"] for summary in camera_summaries)
        total_fn = sum(summary["fn"] for summary in camera_summaries)
        overall = {
            "model": args.model,
            "foot_mode": args.foot_mode,
            "cameras": [summary["camera"] for summary in camera_summaries],
            "frames": sum(summary["frames"] for summary in camera_summaries),
            "saved_images": sum(summary["saved_images"] for summary in camera_summaries),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": total_tp / max(total_tp + total_fp, 1),
            "recall": total_tp / max(total_tp + total_fn, 1),
            "mean_foot_error_px": float(np.mean(all_foot_errors)) if all_foot_errors else None,
            "median_foot_error_px": float(np.median(all_foot_errors)) if all_foot_errors else None,
            "per_camera": [
                {key: value for key, value in summary.items() if key != "per_frame"}
                for summary in camera_summaries
            ],
        }
        write_reports(output, overall, all_csv_rows, args)
        print("Overall:")
        print(json.dumps({key: value for key, value in overall.items() if key != "per_camera"}, indent=2))

    print(f"Results written to: {output}")


if __name__ == "__main__":
    main()
