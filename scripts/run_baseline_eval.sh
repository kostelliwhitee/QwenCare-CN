#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PROJECT_PYTHONPATH}"

MODEL_PATH="${1:-models/Qwen3-8B}"
OUT_DIR="${2:-results/baseline_qwen3}"

mkdir -p "${OUT_DIR}"

"${CONDA_ENV_PYTHON}" evaluation/run_batch_inference.py \
  --model-name-or-path "${MODEL_PATH}" \
  --input-file evaluation/baseline_prompts.json \
  --output-file "${OUT_DIR}/responses.json"

"${CONDA_ENV_PYTHON}" evaluation/compute_response_metrics.py \
  --prediction-file "${OUT_DIR}/responses.json" \
  --output-file "${OUT_DIR}/metrics.json"

"${CONDA_ENV_PYTHON}" evaluation/score_simulated_dialogues.py \
  --prediction-file "${OUT_DIR}/responses.json" \
  --output-file "${OUT_DIR}/scenario_scores.json"
