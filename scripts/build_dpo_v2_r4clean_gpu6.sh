#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PROJECT_PYTHONPATH}"

SEED_FILE="${1:-data/processed/dpo_seed_prompts.json}"
SFT_MODEL_PATH="${2:-runs/sft_qwen3_v2_r4_clean_gpu4}"
OUTPUT_DIR="${3:-data/processed/v2_r4_clean_dpo}"

mkdir -p "${OUTPUT_DIR}" logs

"${CONDA_ENV_PYTHON}" training/generate_dpo_candidates.py \
  --model-name-or-path models/Qwen3-8B \
  --input-file "${SEED_FILE}" \
  --output-file "${OUTPUT_DIR}/dpo_candidates_baseline_v2.json" \
  --num-candidates 3 \
  --max-new-tokens 192 \
  --max-attempts-per-prompt 6 \
  --temperatures 0.2,0.6,0.8 \
  --progress-every 25

"${CONDA_ENV_PYTHON}" training/generate_dpo_candidates.py \
  --model-name-or-path "${SFT_MODEL_PATH}" \
  --input-file "${SEED_FILE}" \
  --output-file "${OUTPUT_DIR}/dpo_candidates_sft_v2.json" \
  --num-candidates 3 \
  --max-new-tokens 192 \
  --max-attempts-per-prompt 6 \
  --temperatures 0.2,0.6,0.8 \
  --progress-every 25

"${CONDA_ENV_PYTHON}" training/build_dpo_pairs.py \
  --candidate-file "${SEED_FILE}" \
  --candidate-file "${OUTPUT_DIR}/dpo_candidates_baseline_v2.json" \
  --candidate-file "${OUTPUT_DIR}/dpo_candidates_sft_v2.json" \
  --output-file "${OUTPUT_DIR}/dpo_pairs_v2.json" \
  --min-margin 2.0 \
  --min-usable-candidates 3
