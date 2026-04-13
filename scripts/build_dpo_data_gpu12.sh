#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}" \
  scripts/build_dpo_data.sh \
  "${1:-runs/sft_qwen3_lora}" \
  "${2:-data/processed/dpo_seed_prompts.json}" \
  "${3:-data/processed/dpo_candidates.json}" \
  "${4:-data/processed/dpo_pairs.json}"
