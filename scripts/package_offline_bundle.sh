#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${1:-${PROJECT_ROOT}/dist/offline_bundle}"
TARBALL_PATH="${2:-${PROJECT_ROOT}/dist/emocalcu_offline_bundle.tar.gz}"
INCLUDE_BASE_MODEL="${INCLUDE_BASE_MODEL:-0}"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}" "${PROJECT_ROOT}/dist"

cp -a "${PROJECT_ROOT}/README.md" "${OUTPUT_DIR}/"
cp -a "${PROJECT_ROOT}/PROJECT_STATUS.md" "${OUTPUT_DIR}/"
cp -a "${PROJECT_ROOT}/EXPERIMENT_LOG.md" "${OUTPUT_DIR}/"
cp -a "${PROJECT_ROOT}/environment.yml" "${OUTPUT_DIR}/"
cp -a "${PROJECT_ROOT}/configs" "${OUTPUT_DIR}/"
cp -a "${PROJECT_ROOT}/demo" "${OUTPUT_DIR}/"
cp -a "${PROJECT_ROOT}/deployment" "${OUTPUT_DIR}/"
cp -a "${PROJECT_ROOT}/docs" "${OUTPUT_DIR}/"
cp -a "${PROJECT_ROOT}/evaluation" "${OUTPUT_DIR}/"
cp -a "${PROJECT_ROOT}/reports" "${OUTPUT_DIR}/"
cp -a "${PROJECT_ROOT}/scripts" "${OUTPUT_DIR}/"
cp -a "${PROJECT_ROOT}/training" "${OUTPUT_DIR}/"

mkdir -p "${OUTPUT_DIR}/results"
cp -a "${PROJECT_ROOT}/results/comparison_qwen3_v2_final" "${OUTPUT_DIR}/results/"
cp -a "${PROJECT_ROOT}/results/quick_eval_baseline_q25" "${OUTPUT_DIR}/results/"
cp -a "${PROJECT_ROOT}/results/quick_eval_sft_v2_q25" "${OUTPUT_DIR}/results/"
cp -a "${PROJECT_ROOT}/results/quick_eval_dpo_v2_q25" "${OUTPUT_DIR}/results/"

mkdir -p "${OUTPUT_DIR}/runs"
cp -a "${PROJECT_ROOT}/runs/dpo_qwen3_v2_r4_clean_gpu34" "${OUTPUT_DIR}/runs/"

if [[ "${INCLUDE_BASE_MODEL}" == "1" ]]; then
  mkdir -p "${OUTPUT_DIR}/models"
  cp -a "${PROJECT_ROOT}/models/Qwen3-8B" "${OUTPUT_DIR}/models/"
fi

cat > "${OUTPUT_DIR}/BUNDLE_README.md" <<EOF
# Offline Bundle

This bundle contains:

- full project source
- final evaluation summaries
- final DPO v2 adapter

Optional base-model snapshot included: ${INCLUDE_BASE_MODEL}

Recommended local demo:

\`\`\`bash
CUDA_VISIBLE_DEVICES=1 \\
DEMO_MODEL_PATH=./models/Qwen3-8B \\
DEMO_ADAPTER_PATH=./runs/dpo_qwen3_v2_r4_clean_gpu34 \\
bash scripts/run_demo_cli_hf.sh
\`\`\`
EOF

TMP_TARBALL="$(mktemp /tmp/emocalcu_offline_bundle.XXXXXX.tar.gz)"
tar -czf "${TMP_TARBALL}" -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")"
mv "${TMP_TARBALL}" "${TARBALL_PATH}"
echo "created ${OUTPUT_DIR}"
echo "created ${TARBALL_PATH}"
