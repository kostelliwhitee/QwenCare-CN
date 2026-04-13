#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PROJECT_PYTHONPATH}"

MODEL_PATH="${1:?model path is required}"
PROMPT_FILE="${2:-data/processed/dpo_seed_prompts.json}"
CANDIDATE_FILE="${3:-data/processed/dpo_candidates.json}"
PAIR_FILE="${4:-data/processed/dpo_pairs.json}"

mkdir -p "$(dirname "${CANDIDATE_FILE}")" "$(dirname "${PAIR_FILE}")"

"${CONDA_ENV_PYTHON}" training/generate_dpo_candidates.py \
  --model-name-or-path "${MODEL_PATH}" \
  --input-file "${PROMPT_FILE}" \
  --output-file "${CANDIDATE_FILE}"

"${CONDA_ENV_PYTHON}" training/build_dpo_pairs.py \
  --candidate-file "${CANDIDATE_FILE}" \
  --output-file "${PAIR_FILE}"
