#!/usr/bin/env bash
# Quick entrypoint. Settings below are forwarded to original MVDet in MVDetBRL.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

# Dataset/training paths are resolved after entering MVDetBRL.
export DATASET="${DATASET:-wildtrack}"
case "$DATASET" in
  wildtrack)
    default_data_path="../Data/Wildtrack"
    default_pseudo_dir="./pseudo_labels/wildtrack_yolo26m"
    ;;
  multiviewx)
    default_data_path="../Data/MultiviewX"
    default_pseudo_dir="./pseudo_labels/multiviewx_yolo26m"
    ;;
  *) echo "Unsupported DATASET: $DATASET" >&2; exit 2 ;;
esac
export DATA_PATH="${DATA_PATH:-$default_data_path}"
export PSEUDO_DIR="${PSEUDO_DIR:-$default_pseudo_dir}"
# Optional: export YOLO_MODEL="./custom-pose.pt". Otherwise an existing
# yolo26m-pose.pt is reused, or Ultralytics downloads it automatically.

# Quick experiment settings; override from the shell or edit here.
export DROP_RATIOS="${DROP_RATIOS:-45}"
export LAMBDAS="${LAMBDAS:-0.005 0.01 0.025}"
export SEEDS="${SEEDS:-1}"
export EPOCHS="${EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export LR="${LR:-0.1}"
export MOMENTUM="${MOMENTUM:-0.5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
export ALPHA="${ALPHA:-1.0}"
export BRL_POS_THR="${BRL_POS_THR:-0.1}"
export PSEUDO_THR="${PSEUDO_THR:-0.1}"
export GPU="${GPU:-0}"
export AUTO_PREPARE_PSEUDO="${AUTO_PREPARE_PSEUDO:-1}"
export FORCE_REGENERATE_PSEUDO="${FORCE_REGENERATE_PSEUDO:-0}"
export MIN_PSEUDO_FILES="${MIN_PSEUDO_FILES:-360}"
export YOLO_IMGSZ="${YOLO_IMGSZ:-1280}"
export YOLO_DEVICE="${YOLO_DEVICE:-0}"
export YOLO_CONF="${YOLO_CONF:-0.25}"
export YOLO_KPT_CONF="${YOLO_KPT_CONF:-0.35}"
export YOLO_CANDIDATE_CONF="${YOLO_CANDIDATE_CONF:-0.15}"
export YOLO_FOOT_ANCHOR="${YOLO_FOOT_ANCHOR:-pose_x_bbox_y}"
export YOLO_MIN_VIEWS="${YOLO_MIN_VIEWS:-2}"
export YOLO_MERGE_RADIUS_M="${YOLO_MERGE_RADIUS_M:-0.60}"
export YOLO_TEMPORAL_SINGLETONS="${YOLO_TEMPORAL_SINGLETONS:-1}"
export YOLO_TEMPORAL_CONF="${YOLO_TEMPORAL_CONF:-0.65}"
export YOLO_TEMPORAL_RADIUS_M="${YOLO_TEMPORAL_RADIUS_M:-0.60}"
export YOLO_INFERENCE_BATCH="${YOLO_INFERENCE_BATCH:-4}"

echo "Redirecting to MVDetBRL/train_grid_pseudo_only.sh"
exec bash "$script_dir/../MVDetBRL/train_grid_pseudo_only.sh" "$@"
