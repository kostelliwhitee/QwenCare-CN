#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

exec env \
  PATH="${CONDA_ENV_PREFIX}/bin:${PATH}" \
  PYTHONPATH="${PROJECT_PYTHONPATH}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  HF_HOME="${HF_HOME}" \
  TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  "$@"
