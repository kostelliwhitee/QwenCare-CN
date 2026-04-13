# QwenCare-CN 最终交付说明

## 1. 当前最终模型

- 默认最终模型：`runs/dpo_qwen3_v2_r4_clean_gpu34`
- 选择依据：在当前同口径 `quick_eval_q25` 上，`DPO v2` 的 `avg_overall`、`avg_supportiveness`、`avg_safety` 和 `junk_rate` 都优于 baseline 与 `SFT v2`

## 2. 最终结果摘要

| model | junk_rate | avg_supportiveness | avg_safety | avg_overall |
| --- | --- | --- | --- | --- |
| baseline_q25 | 0.16 | 2.08 | 4.56 | 2.76 |
| sft_v2_q25 | 0.08 | 2.20 | 4.84 | 3.12 |
| dpo_v2_q25 | 0.04 | 2.48 | 4.92 | 3.44 |

完整表格见 [results/comparison_qwen3_v2_final/comparison_summary.md](/9950backfile/heruxuan/emocalcu/results/comparison_qwen3_v2_final/comparison_summary.md)。

## 3. 代码仓库最终结构

```text
.
├── README.md
├── environment.yml
├── configs/
├── data/
├── demo/
├── deployment/
├── docs/
├── evaluation/
├── logs/
├── reports/
├── results/
├── runs/
├── scripts/
├── training/
├── PROJECT_STATUS.md
└── EXPERIMENT_LOG.md
```

关键正式产物：

- 正式 `SFT v1`：`runs/sft_qwen3_lora`
- 正式 `DPO v1`：`runs/dpo_qwen3_baseline_pairs_gpu12`
- 正式 `SFT v2`：`runs/sft_qwen3_v2_r4_clean_gpu4`
- 正式 `DPO v2`：`runs/dpo_qwen3_v2_r4_clean_gpu34`

## 4. 关键启动命令

### 数据准备

```bash
python3 scripts/prepare_smile_data.py \
  --input-dir data/raw/smile-main/data \
  --output-dir data/processed
```

### `SFT v2`

```bash
CUDA_VISIBLE_DEVICES=4 bash scripts/run_sft_v2_r4_clean_gpu4.sh
```

### `DPO v2`

```bash
CUDA_VISIBLE_DEVICES=3,4 bash scripts/run_dpo_v2_gpu56.sh \
  data/processed/v2_r4_clean_dpo/dpo_pairs_v2.json \
  runs/sft_qwen3_v2_r4_clean_gpu4 \
  runs/dpo_qwen3_v2_r4_clean_gpu34
```

### 最终对比汇总

```bash
python3 evaluation/build_comparison_summary.py \
  --results \
  baseline_q25=results/quick_eval_baseline_q25 \
  sft_v2_q25=results/quick_eval_sft_v2_q25 \
  dpo_v2_q25=results/quick_eval_dpo_v2_q25 \
  --output-dir results/comparison_qwen3_v2_final
```

## 5. Demo 启动方式

### 推荐：最终模型 CLI

```bash
CUDA_VISIBLE_DEVICES=1 \
DEMO_MODEL_PATH=/9950backfile/heruxuan/emocalcu/models/Qwen3-8B \
DEMO_ADAPTER_PATH=/9950backfile/heruxuan/emocalcu/runs/dpo_qwen3_v2_r4_clean_gpu34 \
bash scripts/run_demo_cli_hf.sh
```

### 推荐：最终模型 Gradio

```bash
CUDA_VISIBLE_DEVICES=1,2 \
DEMO_MODEL_PATH=/9950backfile/heruxuan/emocalcu/models/Qwen3-8B \
DEMO_ADAPTER_PATH=/9950backfile/heruxuan/emocalcu/runs/dpo_qwen3_v2_r4_clean_gpu34 \
bash scripts/run_demo_hf_gpu12.sh
```

### 已验证：基座模型 `vLLM + Gradio`

```bash
CUDA_VISIBLE_DEVICES=5 \
TENSOR_PARALLEL_SIZE=1 \
GPU_MEMORY_UTILIZATION=0.75 \
MAX_MODEL_LEN=8192 \
ENFORCE_EAGER=1 \
PORT=8001 \
deployment/serve_vllm.sh models/Qwen3-8B
```

```bash
VLLM_PORT=8001 CUDA_VISIBLE_DEVICES=1,2 bash scripts/run_demo_vllm_gpu12.sh
```

说明：当前已验证的 `vLLM` 路线服务对象是基座模型，不是最终 `DPO v2` adapter。

## 6. 报告位置

- 最终报告：[reports/final_report.md](/9950backfile/heruxuan/emocalcu/reports/final_report.md)
- 状态文档：[PROJECT_STATUS.md](/9950backfile/heruxuan/emocalcu/PROJECT_STATUS.md)
- 实验记录：[EXPERIMENT_LOG.md](/9950backfile/heruxuan/emocalcu/EXPERIMENT_LOG.md)

## 7. 尚未解决的问题

- `MT-Bench/MMLU` 尚未补跑，因此通用能力保持还没有独立 benchmark 证据
- `DPO v2` 的 `distinct-1/2` 低于 baseline，多样性没有同步提升
- `avg_exploration` 没有明显提高，模型更偏“支持性”而不是“探索式”回复
- 当前 `vLLM + Gradio` 的稳定验证对象是基座模型，最终 adapter 的服务化路径仍待补强

## 8. 后续可优化方向

- 补跑 `MT-Bench/MMLU` 子集
- 扩展更大规模 held-out 场景评测
- 单独增强“探索式提问”行为的数据和偏好对
- 引入更强的本地 judge，扩大 `DPO v2` pair 规模
- 进一步研究最终 adapter 与 `vLLM` 的服务化整合

## 9. 打包与上传准备

- GitHub 源码包：
  - `bash scripts/package_github_source.sh`
- Hugging Face 模型包：
  - `bash scripts/package_hf_model_bundle.sh`
- 离线整包：
  - `bash scripts/package_offline_bundle.sh`

详细说明见 [publish_guide.md](/9950backfile/heruxuan/emocalcu/docs/publish_guide.md)。
