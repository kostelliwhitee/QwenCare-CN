#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${1:-${PROJECT_ROOT}/dist/github_source_bundle}"
TARBALL_PATH="${2:-${PROJECT_ROOT}/dist/emocalcu_github_source.tar.gz}"

rm -rf "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${PROJECT_ROOT}/dist"

copy_into_bundle() {
  local src="$1"
  local dst_dir="$2"
  mkdir -p "${dst_dir}"
  cp -a "${src}" "${dst_dir}/"
}

copy_into_bundle "${PROJECT_ROOT}/README.md" "${OUTPUT_ROOT}"
copy_into_bundle "${PROJECT_ROOT}/PROJECT_STATUS.md" "${OUTPUT_ROOT}"
copy_into_bundle "${PROJECT_ROOT}/EXPERIMENT_LOG.md" "${OUTPUT_ROOT}"
copy_into_bundle "${PROJECT_ROOT}/environment.yml" "${OUTPUT_ROOT}"
copy_into_bundle "${PROJECT_ROOT}/.gitignore" "${OUTPUT_ROOT}"
copy_into_bundle "${PROJECT_ROOT}/configs" "${OUTPUT_ROOT}"
copy_into_bundle "${PROJECT_ROOT}/demo" "${OUTPUT_ROOT}"
copy_into_bundle "${PROJECT_ROOT}/deployment" "${OUTPUT_ROOT}"
copy_into_bundle "${PROJECT_ROOT}/docs" "${OUTPUT_ROOT}"
copy_into_bundle "${PROJECT_ROOT}/evaluation" "${OUTPUT_ROOT}"
copy_into_bundle "${PROJECT_ROOT}/reports" "${OUTPUT_ROOT}"
copy_into_bundle "${PROJECT_ROOT}/scripts" "${OUTPUT_ROOT}"
copy_into_bundle "${PROJECT_ROOT}/training" "${OUTPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}/results"
copy_into_bundle "${PROJECT_ROOT}/results/comparison_qwen3" "${OUTPUT_ROOT}/results"
copy_into_bundle "${PROJECT_ROOT}/results/comparison_qwen3_v2_final" "${OUTPUT_ROOT}/results"
copy_into_bundle "${PROJECT_ROOT}/results/quick_eval_baseline_q25" "${OUTPUT_ROOT}/results"
copy_into_bundle "${PROJECT_ROOT}/results/quick_eval_sft_v2_q25" "${OUTPUT_ROOT}/results"
copy_into_bundle "${PROJECT_ROOT}/results/quick_eval_dpo_v2_q25" "${OUTPUT_ROOT}/results"
copy_into_bundle "${PROJECT_ROOT}/results/v2_r4_clean_audit_report.json" "${OUTPUT_ROOT}/results"

mkdir -p "${OUTPUT_ROOT}/data/processed/v2_r4_clean"
copy_into_bundle "${PROJECT_ROOT}/data/processed/dataset_stats.json" "${OUTPUT_ROOT}/data/processed"
copy_into_bundle "${PROJECT_ROOT}/data/processed/v2_r4_clean/train_clean_summary.json" "${OUTPUT_ROOT}/data/processed/v2_r4_clean"
copy_into_bundle "${PROJECT_ROOT}/data/processed/v2_r4_clean/validation_clean_summary.json" "${OUTPUT_ROOT}/data/processed/v2_r4_clean"

cat > "${OUTPUT_ROOT}/RELEASE_MANIFEST.md" <<'EOF'
# Release Manifest

This source bundle contains:

- project code
- final report and delivery docs
- lightweight evaluation summaries
- dataset statistics and final cleaned data summaries

This bundle does not contain:

- raw datasets
- full processed training data
- full base model weights
- training logs

To package the final model adapter for Hugging Face, run:

```bash
bash scripts/package_hf_model_bundle.sh
```
EOF

tar -czf "${TARBALL_PATH}" -C "$(dirname "${OUTPUT_ROOT}")" "$(basename "${OUTPUT_ROOT}")"
echo "created ${OUTPUT_ROOT}"
echo "created ${TARBALL_PATH}"
