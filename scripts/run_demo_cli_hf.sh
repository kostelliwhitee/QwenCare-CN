#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PYTHONPATH}"
export DEMO_BACKEND="${DEMO_BACKEND:-hf}"
export DEMO_MODEL_PATH="${DEMO_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-8B}"

exec "${CONDA_ENV_PYTHON}" "${PROJECT_ROOT}/demo/cli_chat.py" "$@"
