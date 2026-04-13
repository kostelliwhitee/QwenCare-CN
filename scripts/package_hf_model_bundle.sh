#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${1:-${PROJECT_ROOT}/runs/dpo_qwen3_v2_r4_clean_gpu34}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/dist/hf_dpo_v2_model}"
TARBALL_PATH="${3:-${PROJECT_ROOT}/dist/emocalcu_dpo_v2_hf_model.tar.gz}"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}" "${PROJECT_ROOT}/dist"

cp -a "${MODEL_DIR}/." "${OUTPUT_DIR}/"
mkdir -p "${OUTPUT_DIR}/eval"
cp -a "${PROJECT_ROOT}/results/comparison_qwen3_v2_final/comparison_summary.md" "${OUTPUT_DIR}/eval/"
cp -a "${PROJECT_ROOT}/results/quick_eval_dpo_v2_q25/metrics.json" "${OUTPUT_DIR}/eval/"
cp -a "${PROJECT_ROOT}/results/quick_eval_dpo_v2_q25/scenario_scores.json" "${OUTPUT_DIR}/eval/"

cat > "${OUTPUT_DIR}/.gitattributes" <<'EOF'
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
EOF

cat > "${OUTPUT_DIR}/README.md" <<'EOF'
# emocalcu-dpo-v2-qwen3-8b

This repository contains the final LoRA adapter from the course project:

- Project: QwenCare-CN
- Base model: `Qwen/Qwen3-8B`
- Adapter type: LoRA
- Final adapter path in local project: `runs/dpo_qwen3_v2_r4_clean_gpu34`

## Intended use

This adapter is intended for:

- Chinese emotional support dialogue
- course-project research demos
- non-diagnostic supportive conversation experiments

It is not a medical device and must not be used as a substitute for professional diagnosis or treatment.

## Final quick-eval result

Under the project-internal `quick_eval_q25` setting:

| model | junk_rate | avg_supportiveness | avg_safety | avg_overall |
| --- | --- | --- | --- | --- |
| baseline_q25 | 0.16 | 2.08 | 4.56 | 2.76 |
| sft_v2_q25 | 0.08 | 2.20 | 4.84 | 3.12 |
| dpo_v2_q25 | 0.04 | 2.48 | 4.92 | 3.44 |

## Load with PEFT

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model_path = "Qwen/Qwen3-8B"
adapter_path = "./"

tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
```

## Notes

- The conclusion above is based on project-internal proxy evaluation and scenario scoring.
- `MT-Bench/MMLU` were not fully rerun in this round, so general-capability retention is not independently benchmarked yet.
EOF

tar -czf "${TARBALL_PATH}" -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")"
echo "created ${OUTPUT_DIR}"
echo "created ${TARBALL_PATH}"
