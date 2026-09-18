#!/usr/bin/env bash
# SHOTBRL — BRL v1 on dropped annotations (no full / drop_0).
set -euo pipefail

cd "$(dirname "$0")"

WILDTRACK_ROOT="${WILDTRACK_ROOT:-/kaggle/input/datasets/lee735/thesis-dataset-new/Wildtrack/Wildtrack}"
MULTIVIEWX_ROOT="${MULTIVIEWX_ROOT:-/kaggle/input/datasets/lee735/thesis-dataset-new/MultiviewX/MultiviewX}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/shotbrl-output}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
DATASETS="${DATASETS:-wildtrack multiviewx}"
DROP_RATIOS="${DROP_RATIOS:-20 45 60}"

mkdir -p "$OUTPUT_ROOT"

for dataset in $DATASETS; do
  case "$dataset" in
    wildtrack) data_root="$WILDTRACK_ROOT" ;;
    multiviewx) data_root="$MULTIVIEWX_ROOT" ;;
    *)
      echo "ERROR: Unsupported dataset: $dataset" >&2
      exit 1
      ;;
  esac

  if [[ ! -d "$data_root" ]]; then
    echo "ERROR: Dataset root not found: $data_root" >&2
    exit 1
  fi

  for drop_ratio in $DROP_RATIOS; do
    annotation_root="$data_root/drop_annotations/drop_${drop_ratio}/annotations_positions"
    if [[ ! -d "$annotation_root" ]]; then
      echo "ERROR: Dropped annotations not found: $annotation_root" >&2
      exit 1
    fi

    echo "===== Running $dataset / brl / drop_ratio=$drop_ratio ====="
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u main.py \
      -d "$dataset" \
      --data_root "$data_root" \
      --output_root "$OUTPUT_ROOT" \
      --drop_ratio "$drop_ratio" \
      --loss brl
  done
done
echo "===== All done ====="
