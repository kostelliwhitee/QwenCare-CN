#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PROJECT_PYTHONPATH}"

MODEL_PATH="${1:?model path is required}"
OUT_DIR="${2:?output dir is required}"
PROMPT_FILE="${3:-evaluation/extended_eval_prompts.json}"

mkdir -p "${OUT_DIR}"

"${CONDA_ENV_PYTHON}" evaluation/run_batch_inference.py \
  --model-name-or-path "${MODEL_PATH}" \
  --input-file "${PROMPT_FILE}" \
  --output-file "${OUT_DIR}/responses.json"

"${CONDA_ENV_PYTHON}" evaluation/compute_response_metrics.py \
  --prediction-file "${OUT_DIR}/responses.json" \
  --output-file "${OUT_DIR}/metrics.json"

"${CONDA_ENV_PYTHON}" evaluation/score_simulated_dialogues.py \
  --prediction-file "${OUT_DIR}/responses.json" \
  --output-file "${OUT_DIR}/scenario_scores.json"
