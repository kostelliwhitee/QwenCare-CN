#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export PYTHONPATH="${PROJECT_PYTHONPATH}"
export DEMO_BACKEND="vllm"
export DEMO_MODEL_PATH="${DEMO_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-8B}"
export VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:${VLLM_PORT:-8000}/v1}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

exec "${CONDA_ENV_PYTHON}" "${PROJECT_ROOT}/demo/app.py"
