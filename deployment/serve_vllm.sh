#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
MODEL_PATH="${1:-${PROJECT_ROOT}/models/Qwen3-8B}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

if ! "${CONDA_ENV_PYTHON}" -c "import vllm" >/dev/null 2>&1; then
  echo "vLLM is not installed in the current environment. Please install it first or use the HF Gradio fallback demo." >&2
  exit 1
fi

CMD=(
  "${CONDA_BIN}" run -n "${CONDA_ENV_NAME}" python -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --host 0.0.0.0
  --port "${PORT}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype bfloat16
)

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  CMD+=(--enforce-eager)
fi

if [[ -n "${MAX_MODEL_LEN}" ]]; then
  CMD+=(--max-model-len "${MAX_MODEL_LEN}")
fi

exec "${CMD[@]}"
