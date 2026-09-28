"""Run a small YOLO pose foot-point demo and compare it with Wildtrack boxes.

The script intentionally keeps model inference outside the MVDet training loop.  It
produces annotated images plus JSON/CSV summaries that can be inspected before
generating BEV pseudo labels for the full training set.
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
    parser = argparse.ArgumentParser(description="YOLO pose foot demo for Wildtrack")
    parser.add_argument("--source", type=Path, required=True, help="Camera image directory, e.g. Image_subsets/C1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo26m-pose.pt")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--kpt-conf", type=float, default=0.35)
    parser.add_argument("--iou-match", type=float, default=0.30)
    parser.add_argument("--device", default="0")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Wildtrack annotations_positions. Defaults to <dataset root>/annotations_positions.",
    )
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


def extract_predictions(result, kpt_conf):
    if result.boxes is None or result.keypoints is None:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    box_scores = result.boxes.conf.detach().cpu().numpy()
    keypoints = result.keypoints.data.detach().cpu().numpy()
    predictions = []
    for box, box_score, pose in zip(boxes, box_scores, keypoints):
        left, right = pose[LEFT_ANKLE], pose[RIGHT_ANKLE]
        valid_left = left.shape[0] >= 3 and float(left[2]) >= kpt_conf
        valid_right = right.shape[0] >= 3 and float(right[2]) >= kpt_conf
        if valid_left and valid_right:
            foot = (left[:2] + right[:2]) / 2.0
            foot_conf = float((left[2] + right[2]) / 2.0)
            source = "ankle_midpoint"
        elif valid_left or valid_right:
            ankle = left if valid_left else right
            bbox_bottom = np.asarray([(box[0] + box[2]) / 2.0, box[3]], dtype=np.float32)
            foot = 0.5 * ankle[:2] + 0.5 * bbox_bottom
            foot_conf = float(ankle[2]) * 0.75
            source = "single_ankle_bbox"
        else:
            foot = np.asarray([(box[0] + box[2]) / 2.0, box[3]], dtype=np.float32)
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


def main():
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / "Ultralytics").resolve()))
    from ultralytics import YOLO

    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    annotations = args.annotations
    if annotations is None:
        annotations = source.parent.parent / "annotations_positions"

    image_paths = sorted(path for path in source.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    image_paths = image_paths[: args.num_frames]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source}")

    model = YOLO(args.model)
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        verbose=True,
        save=False,
    )

    frame_summaries = []
    csv_rows = []
    total_tp = total_fp = total_fn = 0
    all_foot_errors = []
    for image_path, result in zip(image_paths, results):
        frame = int(image_path.stem)
        predictions = extract_predictions(result, args.kpt_conf)
        ground_truth = load_gt(annotations / f"{image_path.stem}.json", args.camera_index)
        matches, unmatched_pred, unmatched_gt = match_predictions(predictions, ground_truth, args.iou_match)
        total_tp += len(matches)
        total_fp += len(unmatched_pred)
        total_fn += len(unmatched_gt)
        all_foot_errors.extend(match[3] for match in matches)

        image = cv2.imread(str(image_path))
        annotated = draw_frame(image, predictions, ground_truth, matches, unmatched_pred, unmatched_gt)
        cv2.imwrite(str(output / f"{image_path.stem}_pose_foot.jpg"), annotated)

        frame_summary = {
            "frame": frame,
            "image": str(image_path),
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
            csv_rows.append({"frame": frame, **record})
        frame_summaries.append(frame_summary)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    summary = {
        "model": args.model,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "keypoint_conf": args.kpt_conf,
        "iou_match": args.iou_match,
        "frames": len(image_paths),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "mean_foot_error_px": float(np.mean(all_foot_errors)) if all_foot_errors else None,
        "median_foot_error_px": float(np.median(all_foot_errors)) if all_foot_errors else None,
        "per_frame": frame_summaries,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["frame", "box", "foot", "box_conf", "foot_conf", "confidence", "source", "matched_gt", "iou", "foot_error_px"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "per_frame"}, indent=2))
    print(f"Annotated frames and metrics written to: {output}")


if __name__ == "__main__":
    main()
