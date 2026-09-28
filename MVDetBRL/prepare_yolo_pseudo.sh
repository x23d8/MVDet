#!/usr/bin/env bash
# Ensure YOLO-pose weights and the complete offline BEV pseudo-label set exist.
set -euo pipefail

cd "$(dirname "$0")"

# Ultralytics appends its own subdirectory; point the base at the writable project.
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$PWD}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
mkdir -p "$YOLO_CONFIG_DIR"

DATASET="${DATASET:-wildtrack}"
DATA_PATH="${DATA_PATH:?DATA_PATH must be set by the training entrypoint}"
PSEUDO_DIR="${PSEUDO_DIR:?PSEUDO_DIR must be set by the training entrypoint}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
MIN_PSEUDO_FILES="${MIN_PSEUDO_FILES:-360}"
AUTO_PREPARE_PSEUDO="${AUTO_PREPARE_PSEUDO:-1}"
FORCE_REGENERATE_PSEUDO="${FORCE_REGENERATE_PSEUDO:-0}"

YOLO_TOOL="${YOLO_TOOL:-../SHOTBRL/tools/generate_yolo_pose_bev_pseudo.py}"
YOLO_MODEL="${YOLO_MODEL:-./yolo26m-pose.pt}"
YOLO_IMGSZ="${YOLO_IMGSZ:-1280}"
YOLO_DEVICE="${YOLO_DEVICE:-0}"
YOLO_CONF="${YOLO_CONF:-0.25}"
YOLO_KPT_CONF="${YOLO_KPT_CONF:-0.35}"
YOLO_CANDIDATE_CONF="${YOLO_CANDIDATE_CONF:-0.15}"
YOLO_FOOT_ANCHOR="${YOLO_FOOT_ANCHOR:-pose_x_bbox_y}"
YOLO_MIN_VIEWS="${YOLO_MIN_VIEWS:-2}"
YOLO_MERGE_RADIUS_M="${YOLO_MERGE_RADIUS_M:-0.60}"
YOLO_TEMPORAL_SINGLETONS="${YOLO_TEMPORAL_SINGLETONS:-1}"
YOLO_TEMPORAL_CONF="${YOLO_TEMPORAL_CONF:-0.65}"
YOLO_TEMPORAL_RADIUS_M="${YOLO_TEMPORAL_RADIUS_M:-0.60}"
YOLO_INFERENCE_BATCH="${YOLO_INFERENCE_BATCH:-4}"

fail() {
  echo "Configuration error: $*" >&2
  exit 2
}

require_positive_int() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer, got '$value'"
}

require_probability() {
  local name="$1" value="$2"
  "$PYTHON_BIN" -c 'import sys; x=float(sys.argv[1]); raise SystemExit(0 if 0.0 <= x <= 1.0 else 1)' "$value" \
    || fail "$name must be in [0, 1], got '$value'"
}

count_pseudo_files() {
  local files
  shopt -s nullglob
  files=("$PSEUDO_DIR"/[0-9]*.json)
  shopt -u nullglob
  echo "${#files[@]}"
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python executable not found: $PYTHON_BIN"
[[ -d "$DATA_PATH" ]] || fail "dataset directory not found: $DATA_PATH"
[[ -f "$YOLO_TOOL" ]] || fail "pseudo-label generator not found: $YOLO_TOOL"
require_positive_int MIN_PSEUDO_FILES "$MIN_PSEUDO_FILES"
require_positive_int YOLO_IMGSZ "$YOLO_IMGSZ"
require_positive_int YOLO_MIN_VIEWS "$YOLO_MIN_VIEWS"
require_positive_int YOLO_INFERENCE_BATCH "$YOLO_INFERENCE_BATCH"
require_probability YOLO_CONF "$YOLO_CONF"
require_probability YOLO_KPT_CONF "$YOLO_KPT_CONF"
require_probability YOLO_CANDIDATE_CONF "$YOLO_CANDIDATE_CONF"
require_probability YOLO_TEMPORAL_CONF "$YOLO_TEMPORAL_CONF"
"$PYTHON_BIN" -c 'import sys; x=float(sys.argv[1]); raise SystemExit(0 if x > 0.0 else 1)' "$YOLO_MERGE_RADIUS_M" \
  || fail "YOLO_MERGE_RADIUS_M must be > 0, got '$YOLO_MERGE_RADIUS_M'"
"$PYTHON_BIN" -c 'import sys; x=float(sys.argv[1]); raise SystemExit(0 if x > 0.0 else 1)' "$YOLO_TEMPORAL_RADIUS_M" \
  || fail "YOLO_TEMPORAL_RADIUS_M must be > 0, got '$YOLO_TEMPORAL_RADIUS_M'"
[[ "$AUTO_PREPARE_PSEUDO" =~ ^[01]$ ]] || fail "AUTO_PREPARE_PSEUDO must be 0 or 1"
[[ "$FORCE_REGENERATE_PSEUDO" =~ ^[01]$ ]] || fail "FORCE_REGENERATE_PSEUDO must be 0 or 1"
[[ "$YOLO_TEMPORAL_SINGLETONS" =~ ^[01]$ ]] || fail "YOLO_TEMPORAL_SINGLETONS must be 0 or 1"
case "$YOLO_FOOT_ANCHOR" in
  pose|bbox_bottom|pose_x_bbox_y) ;;
  *) fail "YOLO_FOOT_ANCHOR must be pose, bbox_bottom, or pose_x_bbox_y" ;;
esac
case "$DATASET" in
  wildtrack) NUM_CAMERAS=7 ;;
  multiviewx) NUM_CAMERAS=6 ;;
  *) fail "unsupported dataset '$DATASET'" ;;
esac
(( YOLO_MIN_VIEWS <= NUM_CAMERAS )) \
  || fail "YOLO_MIN_VIEWS=$YOLO_MIN_VIEWS exceeds the $NUM_CAMERAS available cameras"

mkdir -p "$PSEUDO_DIR"
pseudo_count="$(count_pseudo_files)"
if (( pseudo_count >= MIN_PSEUDO_FILES )) && [[ "$FORCE_REGENERATE_PSEUDO" == "0" ]]; then
  echo "Pseudo labels ready: $pseudo_count JSON files in $PSEUDO_DIR"
  exit 0
fi

if [[ "$AUTO_PREPARE_PSEUDO" != "1" ]]; then
  fail "found $pseudo_count pseudo JSON files; need at least $MIN_PSEUDO_FILES"
fi

echo "Pseudo labels are missing or incomplete ($pseudo_count/$MIN_PSEUDO_FILES)."
"$PYTHON_BIN" -c 'import ultralytics' \
  || fail "Ultralytics is not installed; install the project requirements before retrying"
if [[ ! -f "$YOLO_MODEL" ]]; then
  echo "YOLO model not found at $YOLO_MODEL; downloading it through Ultralytics..."
  mkdir -p "$(dirname "$YOLO_MODEL")"
  "$PYTHON_BIN" -c 'import sys; from ultralytics import YOLO; YOLO(sys.argv[1])' "$YOLO_MODEL"
fi
[[ -f "$YOLO_MODEL" ]] || fail "Ultralytics did not create the expected model file: $YOLO_MODEL"

generator=(
  "$PYTHON_BIN" -u "$YOLO_TOOL"
  --dataset "$DATASET"
  --data-root "$DATA_PATH"
  --output "$PSEUDO_DIR"
  --model "$YOLO_MODEL"
  --imgsz "$YOLO_IMGSZ"
  --conf "$YOLO_CONF"
  --kpt-conf "$YOLO_KPT_CONF"
  --candidate-conf "$YOLO_CANDIDATE_CONF"
  --foot-anchor "$YOLO_FOOT_ANCHOR"
  --min-views "$YOLO_MIN_VIEWS"
  --merge-radius-m "$YOLO_MERGE_RADIUS_M"
  --inference-batch "$YOLO_INFERENCE_BATCH"
  --device "$YOLO_DEVICE"
)
if [[ "$YOLO_TEMPORAL_SINGLETONS" == "1" ]]; then
  generator+=(
    --temporal-singletons
    --temporal-conf "$YOLO_TEMPORAL_CONF"
    --temporal-radius-m "$YOLO_TEMPORAL_RADIUS_M"
  )
fi
if [[ "$FORCE_REGENERATE_PSEUDO" == "1" ]]; then
  generator+=(--overwrite)
fi

echo "Generating pseudo labels with: model=$YOLO_MODEL anchor=$YOLO_FOOT_ANCHOR min_views=$YOLO_MIN_VIEWS merge_radius_m=$YOLO_MERGE_RADIUS_M temporal=$YOLO_TEMPORAL_SINGLETONS"
CUDA_VISIBLE_DEVICES="$GPU" "${generator[@]}"

pseudo_count="$(count_pseudo_files)"
if (( pseudo_count < MIN_PSEUDO_FILES )); then
  fail "generation finished with $pseudo_count JSON files; expected at least $MIN_PSEUDO_FILES"
fi
echo "Pseudo-label preparation complete: $pseudo_count JSON files."
