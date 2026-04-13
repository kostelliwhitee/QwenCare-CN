#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

mkdir -p models logs
export HF_HUB_DISABLE_XET=1

exec "${CONDA_ENV_PYTHON}" -u -c "from huggingface_hub import snapshot_download; path = snapshot_download(repo_id='Qwen/Qwen3-8B', local_dir='models/Qwen3-8B', allow_patterns=['*.json','*.model','*.safetensors','*.py','*.txt','*.md','tokenizer*']); print(path)" \
  2>&1 | tee logs/download_qwen3.log
