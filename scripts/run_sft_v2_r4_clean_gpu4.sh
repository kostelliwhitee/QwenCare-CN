#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PROJECT_PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

TRAIN_FILE="${1:-data/processed/v2_r4_clean/train.json}"
EVAL_FILE="${2:-data/processed/v2_r4_clean/validation.json}"
OUTPUT_DIR="${3:-runs/sft_qwen3_v2_r4_clean_gpu4}"

mkdir -p "${OUTPUT_DIR}" logs

exec "${CONDA_ENV_PYTHON}" -u training/run_sft.py \
  --model-name-or-path models/Qwen3-8B \
  --train-file "${TRAIN_FILE}" \
  --eval-file "${EVAL_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-seq-length 1536 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.05 \
  --num-train-epochs 1.0 \
  --logging-steps 10 \
  --eval-steps 100 \
  --save-steps 100 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  2>&1 | tee logs/sft_qwen3_v2_r4_clean_gpu4.log
