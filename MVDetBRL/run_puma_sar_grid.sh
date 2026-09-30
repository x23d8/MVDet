#!/usr/bin/env bash
set -euo pipefail

# Visibility/scale-biased partial annotations. Generate the split first with:
# python ../simulate_dropped_annotations.py -d Wildtrack -r /path/to/data_parent \
#   --mechanism visibility_sar --seed 1
cd "$(dirname "$0")"

DATASET="${DATASET:-wildtrack}"
DATA_PATH="${DATA_PATH:-/kaggle/working/Data_temp/Wildtrack}"
DEVICES="${DEVICES:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DROP_RATIOS="${DROP_RATIOS:-20 45 60}"
SEEDS="${SEEDS:-1 2 3}"
CLS_THRES="${CLS_THRES:-0.4}"
NMS_RADIUS_M="${NMS_RADIUS_M:-0.3}"
SPLIT_ROOT="${SPLIT_ROOT:-${DATA_PATH}/drop_annotations}"
GT_PATH="${GT_PATH:-$(pwd)/generated_gt/${DATASET}_gt.txt}"

for drop_ratio in ${DROP_RATIOS}; do
  annotation_dir="${SPLIT_ROOT}/visibility_sar/drop_${drop_ratio}/annotations_positions"
  for seed in ${SEEDS}; do
    python main.py \
      --dataset "${DATASET}" --data_path "${DATA_PATH}" \
      --variant puma_hybrid --loss pu --drop_ratio "${drop_ratio}" \
      --pu_propensity_mode sar \
      --train_annotation_dir "${annotation_dir}" \
      --gt_path "${GT_PATH}" \
      --seed "${seed}" --devices "${DEVICES}" --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" --epochs "${EPOCHS}" \
      --optimizer adamw --lr 2e-4 --amp \
      --puma_feature_channels 32 --puma_fused_channels 64 \
      --puma_query_channels 64 --puma_parallel_view_encoding \
      --cls_thres "${CLS_THRES}" --nms_radius_m "${NMS_RADIUS_M}" \
      --skip_initial_test --eval_every 0 \
      --log_interval 10
  done
done
