#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PROJECT_PYTHONPATH}"

TRAIN_FILE="${1:-data/processed/dpo_pairs_baseline.json}"
OUTPUT_DIR="${2:-runs/dpo_qwen3_baseline_pairs}"
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$("${CONDA_ENV_PYTHON}" - <<'PY'
import socket
sock = socket.socket()
sock.bind(("", 0))
print(sock.getsockname()[1])
sock.close()
PY
)"
fi

mkdir -p "${OUTPUT_DIR}" logs

exec "${CONDA_ENV_PREFIX}/bin/torchrun" \
  --nproc_per_node=2 \
  --master_port "${MASTER_PORT}" \
  training/run_dpo.py \
  --base-model-name-or-path models/Qwen3-8B \
  --sft-adapter-path runs/sft_qwen3_lora \
  --train-file "${TRAIN_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 3 \
  --logging-steps 1 \
  --save-steps 10 \
  2>&1 | tee logs/dpo_qwen3.log
