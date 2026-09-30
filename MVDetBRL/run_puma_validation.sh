#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter-development split only. Never report these numbers as test
# results. After selecting settings, retrain with run_puma_grid.sh (0-.9/.9-1).
cd "$(dirname "$0")"

DATASET="${DATASET:-wildtrack}"
DATA_PATH="${DATA_PATH:-/kaggle/working/Data_temp/Wildtrack}"
DEVICES="${DEVICES:-0,1}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DROP_RATIO="${DROP_RATIO:-60}"
FEATURE_CHANNELS="${FEATURE_CHANNELS:-32}"
FUSED_CHANNELS="${FUSED_CHANNELS:-64}"
QUERY_CHANNELS="${QUERY_CHANNELS:-64}"
SPLIT_ROOT="${SPLIT_ROOT:-${DATA_PATH}/drop_annotations}"
GT_PATH="${GT_PATH:-$(pwd)/generated_gt/${DATASET}_gt.txt}"
RESUME_DIR="${RESUME_DIR:-}"
PROPENSITY_MODE="${PROPENSITY_MODE:-scar}"
PARALLEL_VIEW_ENCODING="${PARALLEL_VIEW_ENCODING:-0}"

extra_args=(--gt_path "${GT_PATH}")
if [[ "${DROP_RATIO}" != "0" ]]; then
  annotation_dir="${TRAIN_ANNOTATION_DIR:-${SPLIT_ROOT}/drop_${DROP_RATIO}/annotations_positions}"
  extra_args+=(--train_annotation_dir "${annotation_dir}")
fi
if [[ -n "${RESUME_DIR}" ]]; then
  extra_args+=(--resume "${RESUME_DIR}")
fi
if [[ "${PARALLEL_VIEW_ENCODING}" == "1" ]]; then
  extra_args+=(--puma_parallel_view_encoding)
else
  extra_args+=(--no-puma_parallel_view_encoding)
fi

python main.py \
  --dataset "${DATASET}" --data_path "${DATA_PATH}" \
  --variant puma_hybrid --loss pu --drop_ratio "${DROP_RATIO}" \
  --pu_propensity_mode "${PROPENSITY_MODE}" \
  --train_end_ratio 0.8 --eval_start_ratio 0.8 --eval_end_ratio 0.9 \
  --devices "${DEVICES}" --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" --epochs "${EPOCHS}" \
  --optimizer adamw --lr 2e-4 --amp --no-cudnn_benchmark \
  --puma_feature_channels "${FEATURE_CHANNELS}" \
  --puma_fused_channels "${FUSED_CHANNELS}" \
  --puma_query_channels "${QUERY_CHANNELS}" \
  "${extra_args[@]}" \
  --skip_initial_test --eval_every 0 --save_score_cache --log_interval 10

if [[ -n "${RESUME_DIR}" ]]; then
  cache_path="${RESUME_DIR}/score_cache.npz"
else
  cache_path="$(find "logs/${DATASET}_frame/train_0.8_eval_0.8-0.9" -name score_cache.npz -printf '%T@ %p\n' | sort -nr | sed -n '1p' | cut -d' ' -f2-)"
fi
if [[ ! -f "${cache_path}" ]]; then
  echo "Score cache not found after validation run: ${cache_path}" >&2
  exit 1
fi
python tools/sweep_postprocess.py \
  "${cache_path}" "${GT_PATH}" \
  --output "$(dirname "${cache_path}")/postprocess_sweep.csv"
