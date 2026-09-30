#!/usr/bin/env bash
set -euo pipefail

# Matched-protocol baseline matrix. Legacy MVDet is single-GPU because its
# original implementation hard-codes cuda:0; PUMA uses both GPUs.
cd "$(dirname "$0")"

DATASET="${DATASET:-wildtrack}"
DATA_PATH="${DATA_PATH:-/kaggle/working/Data_temp/Wildtrack}"
DROP_RATIOS="${DROP_RATIOS:-0 20 45 60}"
SEEDS="${SEEDS:-1 2 3}"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SPLIT_ROOT="${SPLIT_ROOT:-${DATA_PATH}/drop_annotations}"
GT_PATH="${GT_PATH:-$(pwd)/generated_gt/${DATASET}_gt.txt}"
PARALLEL_VIEW_ENCODING="${PARALLEL_VIEW_ENCODING:-0}"

for drop_ratio in ${DROP_RATIOS}; do
  for seed in ${SEEDS}; do
    annotation_args=(--gt_path "${GT_PATH}")
    if [[ "${drop_ratio}" != "0" ]]; then
      annotation_args+=(--train_annotation_dir "${SPLIT_ROOT}/drop_${drop_ratio}/annotations_positions")
    fi
    puma_encoding_args=(--no-puma_parallel_view_encoding)
    if [[ "${PARALLEL_VIEW_ENCODING}" == "1" ]]; then
      puma_encoding_args=(--puma_parallel_view_encoding)
    fi
    for loss in mse brl pu; do
      python main.py \
        --dataset "${DATASET}" --data_path "${DATA_PATH}" \
        --variant default --loss "${loss}" --drop_ratio "${drop_ratio}" \
        --seed "${seed}" --devices 0 --batch_size 1 \
        --num_workers "${NUM_WORKERS}" --epochs "${EPOCHS}" \
        "${annotation_args[@]}" \
        --skip_initial_test --eval_every 0 --log_interval 10
    done

    for variant in puma_dense puma_hybrid; do
      python main.py \
        --dataset "${DATASET}" --data_path "${DATA_PATH}" \
        --variant "${variant}" --loss pu --drop_ratio "${drop_ratio}" \
        --seed "${seed}" --devices 0,1 --batch_size 2 \
        --num_workers "${NUM_WORKERS}" --epochs "${EPOCHS}" \
        --optimizer adamw --lr 2e-4 --amp \
        --puma_feature_channels 32 --puma_fused_channels 64 \
        --puma_query_channels 64 "${puma_encoding_args[@]}" --no-cudnn_benchmark \
        "${annotation_args[@]}" \
        --skip_initial_test --eval_every 0 \
        --log_interval 10
    done
  done
done

python tools/summarize_puma_runs.py logs --output logs/benchmark_summary.csv
