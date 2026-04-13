#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export PYTHONPATH="${PROJECT_PYTHONPATH}:${PROJECT_ROOT}"
export DEMO_BACKEND="${DEMO_BACKEND:-hf}"
export DEMO_MODEL_PATH="${DEMO_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-8B}"
export DEMO_WEB_PORT="${DEMO_WEB_PORT:-7861}"

exec "${CONDA_ENV_PYTHON}" -m uvicorn demo.web_app:app --host 0.0.0.0 --port "${DEMO_WEB_PORT}"
