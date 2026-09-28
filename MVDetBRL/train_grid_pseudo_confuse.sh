#!/usr/bin/env bash
# One-command pipeline: prepare YOLO pseudo labels, then train original MVDet.
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Paths (relative to MVDetBRL after the cd above).
DATASET="${DATASET:-wildtrack}"
case "$DATASET" in
  wildtrack)
    DEFAULT_DATA_PATH="../Data/Wildtrack"
    DEFAULT_PSEUDO_DIR="./pseudo_labels/wildtrack_yolo26m"
    ;;
  multiviewx)
    DEFAULT_DATA_PATH="../Data/MultiviewX"
    DEFAULT_PSEUDO_DIR="./pseudo_labels/multiviewx_yolo26m"
    ;;
  *) echo "Unsupported DATASET: $DATASET" >&2; exit 2 ;;
esac
DATA_PATH="${DATA_PATH:-$DEFAULT_DATA_PATH}"
PSEUDO_DIR="${PSEUDO_DIR:-$DEFAULT_PSEUDO_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Training/grid hyperparameters.
DROP_RATIOS="${DROP_RATIOS:-45}"
LAMBDAS="${LAMBDAS:-0.005 0.01 0.025}"
SEEDS="${SEEDS:-1}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-0.1}"
MOMENTUM="${MOMENTUM:-0.5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
ALPHA="${ALPHA:-1.0}"
BRL_POS_THR="${BRL_POS_THR:-0.1}"
BRL_CONFUSE_THR="${BRL_CONFUSE_THR:-0.3}"
BRL_BETA="${BRL_BETA:-0.1}"
PSEUDO_THR="${PSEUDO_THR:-0.1}"
GPU="${GPU:-0}"

# Automatic YOLO/pseudo-label preparation.
MIN_PSEUDO_FILES="${MIN_PSEUDO_FILES:-360}"
AUTO_PREPARE_PSEUDO="${AUTO_PREPARE_PSEUDO:-1}"
FORCE_REGENERATE_PSEUDO="${FORCE_REGENERATE_PSEUDO:-0}"
YOLO_TOOL="${YOLO_TOOL:-../SHOTBRL/tools/generate_yolo_pose_bev_pseudo.py}"
if [[ -z "${YOLO_MODEL:-}" ]]; then
  if [[ -f "../SHOTBRL/yolo26m-pose.pt" ]]; then
    YOLO_MODEL="../SHOTBRL/yolo26m-pose.pt"
  elif [[ -f "../yolo26m-pose.pt" ]]; then
    YOLO_MODEL="../yolo26m-pose.pt"
  else
    YOLO_MODEL="./yolo26m-pose.pt"
  fi
fi
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

fail() { echo "Configuration error: $*" >&2; exit 2; }
positive_int() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "$1 must be a positive integer, got '$2'"; }
nonnegative_int() { [[ "$2" =~ ^[0-9]+$ ]] || fail "$1 must be a non-negative integer, got '$2'"; }
positive_float() {
  "$PYTHON_BIN" -c 'import sys; x=float(sys.argv[1]); raise SystemExit(0 if x > 0 else 1)' "$2" \
    || fail "$1 must be > 0, got '$2'"
}
nonnegative_float() {
  "$PYTHON_BIN" -c 'import sys; x=float(sys.argv[1]); raise SystemExit(0 if x >= 0 else 1)' "$2" \
    || fail "$1 must be >= 0, got '$2'"
}
probability() {
  "$PYTHON_BIN" -c 'import sys; x=float(sys.argv[1]); raise SystemExit(0 if 0 <= x <= 1 else 1)' "$2" \
    || fail "$1 must be in [0, 1], got '$2'"
}
weak_weight() {
  "$PYTHON_BIN" -c 'import sys; x=float(sys.argv[1]); raise SystemExit(0 if 0 < x < 1 else 1)' "$2" \
    || fail "$1 must satisfy 0 < lambda < 1, got '$2'"
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python executable not found: $PYTHON_BIN"
[[ -f main.py ]] || fail "MVDet entrypoint not found: $SCRIPT_DIR/main.py"
[[ -d "$DATA_PATH/Image_subsets" ]] || fail "missing dataset images: $DATA_PATH/Image_subsets"
[[ -d "$DATA_PATH/annotations_positions" ]] || fail "missing full annotations: $DATA_PATH/annotations_positions"
positive_int EPOCHS "$EPOCHS"
positive_int BATCH_SIZE "$BATCH_SIZE"
nonnegative_int NUM_WORKERS "$NUM_WORKERS"
positive_float LR "$LR"
probability MOMENTUM "$MOMENTUM"
nonnegative_float WEIGHT_DECAY "$WEIGHT_DECAY"
nonnegative_float ALPHA "$ALPHA"
probability BRL_POS_THR "$BRL_POS_THR"
probability BRL_CONFUSE_THR "$BRL_CONFUSE_THR"
probability BRL_BETA "$BRL_BETA"
probability PSEUDO_THR "$PSEUDO_THR"
for drop_ratio in $DROP_RATIOS; do
  case "$drop_ratio" in
    0) ;;
    20|45|60)
      [[ -d "$DATA_PATH/drop_annotations/drop_$drop_ratio/annotations_positions" ]] \
        || fail "missing dropped annotations for ratio $drop_ratio"
      ;;
    *) fail "DROP_RATIOS contains unsupported value '$drop_ratio'" ;;
  esac
done
for lambda_pseudo in $LAMBDAS; do weak_weight LAMBDAS "$lambda_pseudo"; done
for seed in $SEEDS; do nonnegative_int SEEDS "$seed"; done

DATASET="$DATASET" DATA_PATH="$DATA_PATH" PSEUDO_DIR="$PSEUDO_DIR" \
PYTHON_BIN="$PYTHON_BIN" GPU="$GPU" MIN_PSEUDO_FILES="$MIN_PSEUDO_FILES" \
AUTO_PREPARE_PSEUDO="$AUTO_PREPARE_PSEUDO" FORCE_REGENERATE_PSEUDO="$FORCE_REGENERATE_PSEUDO" \
YOLO_TOOL="$YOLO_TOOL" YOLO_MODEL="$YOLO_MODEL" YOLO_IMGSZ="$YOLO_IMGSZ" \
YOLO_DEVICE="$YOLO_DEVICE" YOLO_CONF="$YOLO_CONF" YOLO_KPT_CONF="$YOLO_KPT_CONF" \
YOLO_CANDIDATE_CONF="$YOLO_CANDIDATE_CONF" YOLO_FOOT_ANCHOR="$YOLO_FOOT_ANCHOR" \
YOLO_MIN_VIEWS="$YOLO_MIN_VIEWS" YOLO_MERGE_RADIUS_M="$YOLO_MERGE_RADIUS_M" \
YOLO_TEMPORAL_SINGLETONS="$YOLO_TEMPORAL_SINGLETONS" YOLO_TEMPORAL_CONF="$YOLO_TEMPORAL_CONF" \
YOLO_TEMPORAL_RADIUS_M="$YOLO_TEMPORAL_RADIUS_M" \
YOLO_INFERENCE_BATCH="$YOLO_INFERENCE_BATCH" bash "$SCRIPT_DIR/prepare_yolo_pseudo.sh"

echo "===== Model: original MVDet PerspTransDetector ====="
echo "===== Loss: pseudo_confuse ====="
echo "dataset=$DATASET data=$DATA_PATH pseudo=$PSEUDO_DIR model=$YOLO_MODEL"
echo "drop_ratios=[$DROP_RATIOS] lambdas=[$LAMBDAS] seeds=[$SEEDS]"
echo "epochs=$EPOCHS batch=$BATCH_SIZE workers=$NUM_WORKERS lr=$LR momentum=$MOMENTUM weight_decay=$WEIGHT_DECAY alpha=$ALPHA"
echo "brl_pos_thr=$BRL_POS_THR confuse_thr=$BRL_CONFUSE_THR beta=$BRL_BETA pseudo_thr=$PSEUDO_THR"
echo "yolo_conf=$YOLO_CONF kpt_conf=$YOLO_KPT_CONF min_views=$YOLO_MIN_VIEWS merge_radius_m=$YOLO_MERGE_RADIUS_M"
echo "temporal=$YOLO_TEMPORAL_SINGLETONS temporal_conf=$YOLO_TEMPORAL_CONF temporal_radius_m=$YOLO_TEMPORAL_RADIUS_M"

for drop_ratio in $DROP_RATIOS; do
  for lambda_pseudo in $LAMBDAS; do
    for seed in $SEEDS; do
      run_name="mvdet_pseudo_confuse_drop${drop_ratio}_lp${lambda_pseudo}_seed${seed}"
      echo "===== Running $run_name ====="
      command=(
        "$PYTHON_BIN" -u main.py
        -d "$DATASET"
        --data_path "$DATA_PATH"
        --loss brl
        --drop_ratio "$drop_ratio"
        --pseudo_mode pseudo_confuse
        --pseudo_dir "$PSEUDO_DIR"
        --lambda_pseudo "$lambda_pseudo"
        --pseudo_thr "$PSEUDO_THR"
        --brl_pos_thr "$BRL_POS_THR"
        --brl_confuse_thr "$BRL_CONFUSE_THR"
        --brl_beta "$BRL_BETA"
        --brl_no_mirror
        --epochs "$EPOCHS"
        --batch_size "$BATCH_SIZE"
        --num_workers "$NUM_WORKERS"
        --lr "$LR"
        --momentum "$MOMENTUM"
        --weight_decay "$WEIGHT_DECAY"
        --alpha "$ALPHA"
        --seed "$seed"
        --loginfo "$run_name"
      )
      command+=("$@")
      command+=(--variant default)
      CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}"
    done
  done
done

echo "===== MVDet pseudo_confuse grid completed ====="
