#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Reproducible PUMA-MV comparison under complete and controlled SCAR labels.
# Override any value as an environment variable, for example:
#   DATA_PATH=/kaggle/input/.../Wildtrack DEVICES=0,1 BATCH_SIZE=2 ./run_puma_grid.sh

DATA_PATH="${DATA_PATH:-/kaggle/working/Data_temp/Wildtrack}"
DATASET="${DATASET:-wildtrack}"
DEVICES="${DEVICES:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FEATURE_CHANNELS="${FEATURE_CHANNELS:-32}"
FUSED_CHANNELS="${FUSED_CHANNELS:-64}"
QUERY_CHANNELS="${QUERY_CHANNELS:-64}"
LOSS="${LOSS:-pu}"
QUERY_LOSS_WEIGHT="${QUERY_LOSS_WEIGHT:-0.002}"
CLS_THRES="${CLS_THRES:-0.4}"
NMS_RADIUS_M="${NMS_RADIUS_M:-0.3}"
VARIANTS="${VARIANTS:-puma_dense puma_hybrid}"
DROP_RATIOS="${DROP_RATIOS:-0 20 45 60}"
SEEDS="${SEEDS:-1 2 3}"
SPLIT_ROOT="${SPLIT_ROOT:-${DATA_PATH}/drop_annotations}"
GT_PATH="${GT_PATH:-$(pwd)/generated_gt/${DATASET}_gt.txt}"
PARALLEL_VIEW_ENCODING="${PARALLEL_VIEW_ENCODING:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
EVAL_EVERY="${EVAL_EVERY:-1}"

for variant in ${VARIANTS}; do
  for drop_ratio in ${DROP_RATIOS}; do
    for seed in ${SEEDS}; do
      annotation_args=(--gt_path "${GT_PATH}")
      if [[ "${drop_ratio}" != "0" ]]; then
        annotation_args+=(--train_annotation_dir "${SPLIT_ROOT}/drop_${drop_ratio}/annotations_positions")
      fi
      if [[ "${PARALLEL_VIEW_ENCODING}" == "1" ]]; then
        annotation_args+=(--puma_parallel_view_encoding)
      else
        annotation_args+=(--no-puma_parallel_view_encoding)
      fi
      python main.py \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --variant "${variant}" \
        --loss "${LOSS}" \
        --puma_query_loss_weight "${QUERY_LOSS_WEIGHT}" \
        --drop_ratio "${drop_ratio}" \
        --seed "${seed}" \
        --devices "${DEVICES}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --puma_feature_channels "${FEATURE_CHANNELS}" \
        --puma_fused_channels "${FUSED_CHANNELS}" \
        --puma_query_channels "${QUERY_CHANNELS}" \
        "${annotation_args[@]}" \
        --no-cudnn_benchmark \
        --cls_thres "${CLS_THRES}" \
        --nms_radius_m "${NMS_RADIUS_M}" \
        --epochs "${EPOCHS}" \
        --optimizer adamw \
        --lr 2e-4 \
        --amp \
        --skip_initial_test \
        --eval_every "${EVAL_EVERY}" \
        --log_interval "${LOG_INTERVAL}"
    done
  done
done
