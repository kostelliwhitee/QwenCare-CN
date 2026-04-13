#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PROJECT_PYTHONPATH}"

mkdir -p runs/sft_qwen3_lora logs

exec "${CONDA_ENV_PREFIX}/bin/torchrun" \
  --nproc_per_node=2 \
  training/run_sft.py \
  --model-name-or-path models/Qwen3-8B \
  --train-file data/processed/train.json \
  --eval-file data/processed/validation.json \
  --output-dir runs/sft_qwen3_lora \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --num-train-epochs 2 \
  2>&1 | tee logs/sft_qwen3_lora.log
