#!/usr/bin/env bash
# Reproducible positive-unlabeled baseline for controlled missing annotations.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-wildtrack}"
DATA_PATH="${DATA_PATH:-/kaggle/working/Data_temp/Wildtrack}"
DROP_RATIOS="${DROP_RATIOS:-20 45 60}"
SEEDS="${SEEDS:-1 2 3}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU="${GPU:-0}"
PU_POS_THR="${PU_POS_THR:-0.1}"
PU_CLASS_PRIOR="${PU_CLASS_PRIOR:-}"

for drop_ratio in $DROP_RATIOS; do
  for seed in $SEEDS; do
    command=(
      "$PYTHON_BIN" -u main.py
      --dataset "$DATASET"
      --data_path "$DATA_PATH"
      --loss pu
      --drop_ratio "$drop_ratio"
      --pu_pos_thr "$PU_POS_THR"
      --epochs "$EPOCHS"
      --batch_size "$BATCH_SIZE"
      --num_workers "$NUM_WORKERS"
      --seed "$seed"
      --log_interval 10
    )
    if [[ -n "$PU_CLASS_PRIOR" ]]; then
      command+=(--pu_class_prior "$PU_CLASS_PRIOR")
    fi
    echo "===== PU baseline: drop=$drop_ratio seed=$seed ====="
    CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}"
  done
done
