#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PYTHONPATH}"
export DEMO_BACKEND="vllm"
export DEMO_MODEL_PATH="${DEMO_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-8B}"
export VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:${VLLM_PORT:-8001}/v1}"

exec "${CONDA_ENV_PYTHON}" "${PROJECT_ROOT}/demo/cli_chat.py" "$@"
