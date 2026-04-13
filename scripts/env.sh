#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONDA_BIN="/9950backfile/heruxuan/path/miniconda3/bin/conda"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-path}"
export CONDA_ROOT="/9950backfile/heruxuan/path/miniconda3"
if [[ "${CONDA_ENV_NAME}" == "base" ]]; then
  export CONDA_ENV_PREFIX="${CONDA_ROOT}"
else
export CONDA_ENV_PREFIX="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}"
fi
export CONDA_ENV_PYTHON="${CONDA_ENV_PREFIX}/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}"
export PROJECT_PYTHONPATH="${PROJECT_ROOT}/.vendor:${PYTHONPATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_PYTHONPATH}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_ROOT}/.cache/matplotlib}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PROJECT_ROOT}/.cache/triton}"

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${MPLCONFIGDIR}" "${TRITON_CACHE_DIR}"
