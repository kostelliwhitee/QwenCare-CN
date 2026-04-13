#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PROJECT_PYTHONPATH}"

OUTPUT_DIR="${1:-data/processed/v2_r4}"
REWRITE_MODEL_PATH="${2:-models/Qwen3-8B}"
LOG_FILE="${3:-logs/build_sft_v2_data_r4.log}"

mkdir -p "${OUTPUT_DIR}" logs

exec "${CONDA_ENV_PYTHON}" training/prepare_sft_v2_data.py \
  --input-file data/processed/train.json \
  --input-file data/processed/validation.json \
  --output-dir "${OUTPUT_DIR}" \
  --rewrite-model-path "${REWRITE_MODEL_PATH}" \
  --gold-threshold 2.0 \
  --rewrite-threshold 0.0 \
  --rewrite-margin 1.0 \
  --rewrite-limit 10000 \
  --validation-size 1500 \
  --dpo-seed-size 4096 \
  --fallback-keep-threshold 2.0 \
  --progress-every 250 \
  2>&1 | tee "${LOG_FILE}"
