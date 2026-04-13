#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PROJECT_PYTHONPATH}"

INPUT_DIR="${1:-data/processed/v2_r4}"
OUTPUT_DIR="${2:-data/processed/v2_r4_clean}"

mkdir -p "${OUTPUT_DIR}" logs

"${CONDA_ENV_PYTHON}" training/clean_sft_dataset.py \
  --input-file "${INPUT_DIR}/train.json" \
  --output-file "${OUTPUT_DIR}/train.json" \
  --summary-file "${OUTPUT_DIR}/train_clean_summary.json"

"${CONDA_ENV_PYTHON}" training/clean_sft_dataset.py \
  --input-file "${INPUT_DIR}/validation.json" \
  --output-file "${OUTPUT_DIR}/validation.json" \
  --summary-file "${OUTPUT_DIR}/validation_clean_summary.json"
