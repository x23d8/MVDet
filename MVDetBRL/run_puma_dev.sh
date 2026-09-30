#!/usr/bin/env bash
set -euo pipefail

# Fast end-to-end development gate for one-GPU machines. This is deliberately
# too small for reporting accuracy: ten Wildtrack samples train and the next
# ten validate. It exercises PU/query/consistency/checkpoint/evaluator paths
# before committing hours to the official split.
cd "$(dirname "$0")"

DATA_PATH="${DATA_PATH:-/kaggle/working/Data_temp/Wildtrack}"
DEVICES="${DEVICES:-0}"
EPOCHS="${EPOCHS:-1}"
DROP_RATIO="${DROP_RATIO:-0}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-360}"
IMAGE_WIDTH="${IMAGE_WIDTH:-640}"

python main.py \
  --dataset wildtrack --data_path "${DATA_PATH}" \
  --variant puma_hybrid --loss pu --drop_ratio "${DROP_RATIO}" \
  --train_end_ratio 0.025 --eval_start_ratio 0.025 --eval_end_ratio 0.05 \
  --devices "${DEVICES}" --batch_size 1 --num_workers 0 \
  --epochs "${EPOCHS}" --optimizer adamw --lr 2e-4 --amp \
  --image_height "${IMAGE_HEIGHT}" --image_width "${IMAGE_WIDTH}" \
  --puma_feature_channels 16 --puma_fused_channels 48 \
  --puma_query_channels 48 --puma_num_queries 64 \
  --puma_parallel_view_encoding \
  --puma_query_warmup_epochs 0 --puma_query_ramp_epochs 1 \
  --skip_initial_test --eval_every 0 --save_score_cache --log_interval 1
