# QwenCare-CN

`QwenCare-CN` 是一个面向课程项目的完整流水线：中文心理支持数据处理、Qwen3-8B 微调、偏好对齐、自动评测、结果可视化与本地 demo。

## 当前结论

- 已真实完成两轮实验：
  - 第一轮：`baseline / SFT v1 / DPO v1`
  - 第二轮：`SFT v2 / DPO v2`
- 在第一轮小规模评测里，`baseline` 优于早期 `SFT/DPO`。
- 在最新同口径 `quick_eval_q25` 中，当前最优模型已更新为 `DPO v2`。
- 当前默认最终模型：
  - `runs/dpo_qwen3_v2_r4_clean_gpu34`
- 结论边界：
  - 该结论基于当前项目内的自动代理评测与场景评分
  - `MT-Bench/MMLU` 尚未补跑，因此“通用能力不显著下降”还缺独立 benchmark 证据

## 结果概览

### 当前最终对比：`quick_eval_q25`

| model | distinct_1 | distinct_2 | avg_length | junk_rate | avg_supportiveness | avg_safety | avg_overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_q25 | 0.1283 | 0.4721 | 160.28 | 0.16 | 2.08 | 4.56 | 2.76 |
| sft_v2_q25 | 0.1160 | 0.4312 | 195.16 | 0.08 | 2.20 | 4.84 | 3.12 |
| dpo_v2_q25 | 0.1087 | 0.4314 | 197.64 | 0.04 | 2.48 | 4.92 | 3.44 |

完整对比表见 [results/comparison_qwen3_v2_final/comparison_summary.md](/9950backfile/heruxuan/emocalcu/results/comparison_qwen3_v2_final/comparison_summary.md)。

### 历史对比：第一轮正式实验

| model | distinct_1 | distinct_2 | avg_length | avg_overall |
| --- | --- | --- | --- | --- |
| baseline | 0.3333 | 0.7071 | 151.2 | 2.0 |
| sft_v1 | 0.5198 | 0.8056 | 65.8 | 1.0 |
| dpo_v1 | 0.5909 | 0.9474 | 7.3333 | 1.0 |

历史对比表见 [results/comparison_qwen3/comparison_summary.md](/9950backfile/heruxuan/emocalcu/results/comparison_qwen3/comparison_summary.md)。

## 项目结构

```text
.
├── README.md
├── environment.yml
├── configs/
├── data/
│   ├── raw/
│   └── processed/
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

最终交付说明见 [docs/final_delivery.md](/9950backfile/heruxuan/emocalcu/docs/final_delivery.md)。

## 环境准备

默认使用本机 conda 环境：

```bash
scripts/run_in_conda.sh python -V
source scripts/env.sh
```

环境声明见 [environment.yml](/9950backfile/heruxuan/emocalcu/environment.yml)。

当前实验中实际使用过的 GPU 规划：

- baseline / 早期评测 / demo：`GPU 1,2`
- `SFT v2`：`GPU 4`
- `DPO v2`：`GPU 3,4`

## 数据准备

主数据源为 `SmileChat`，基础处理命令：

```bash
python3 scripts/prepare_smile_data.py \
  --input-dir data/raw/smile-main/data \
  --output-dir data/processed
```

第一轮标准切分产物：

- `data/processed/train.json`
- `data/processed/validation.json`
- `data/processed/test.json`
- `data/processed/dataset_stats.json`

第二轮 `v2` 关键数据构建链路：

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/build_sft_v2_data_r4_gpu2.sh
bash scripts/build_v2_r4_clean.sh
```

当前用于正式 `SFT v2` 的训练数据是：

- [train.json](/9950backfile/heruxuan/emocalcu/data/processed/v2_r4_clean/train.json)
- [validation.json](/9950backfile/heruxuan/emocalcu/data/processed/v2_r4_clean/validation.json)

## 训练步骤

### 1. Baseline 评测

```bash
CUDA_VISIBLE_DEVICES=1,2 scripts/run_baseline_eval.sh \
  models/Qwen3-8B \
  results/baseline_qwen3_gpu12
```

### 2. 第一轮正式 SFT / DPO

这部分保留为历史实验链：

```bash
CUDA_VISIBLE_DEVICES=5,6 scripts/run_sft_gpu56.sh
CUDA_VISIBLE_DEVICES=1,2 scripts/build_dpo_data_gpu12.sh
CUDA_VISIBLE_DEVICES=5,6 scripts/run_dpo_gpu56.sh \
  data/processed/dpo_pairs_baseline.json \
  runs/dpo_qwen3_baseline_pairs_gpu12
```

### 3. 第二轮正式 SFT v2

`v2` 的核心改动是：

- `no-think` 序列化
- assistant-only loss
- 更严格的数据过滤与 `baseline rewrite`

正式训练命令：

```bash
CUDA_VISIBLE_DEVICES=4 bash scripts/run_sft_v2_r4_clean_gpu4.sh
```

输出目录：

- [runs/sft_qwen3_v2_r4_clean_gpu4](/9950backfile/heruxuan/emocalcu/runs/sft_qwen3_v2_r4_clean_gpu4)

### 4. 第二轮 DPO v2 数据构建

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/build_dpo_v2_r4clean_gpu6.sh \
  data/processed/dpo_seed_prompts.json \
  runs/sft_qwen3_v2_r4_clean_gpu4 \
  data/processed/v2_r4_clean_dpo
```

关键产物：

- [dpo_pairs_v2.json](/9950backfile/heruxuan/emocalcu/data/processed/v2_r4_clean_dpo/dpo_pairs_v2.json)

### 5. 第二轮正式 DPO v2

```bash
CUDA_VISIBLE_DEVICES=3,4 bash scripts/run_dpo_v2_gpu56.sh \
  data/processed/v2_r4_clean_dpo/dpo_pairs_v2.json \
  runs/sft_qwen3_v2_r4_clean_gpu4 \
  runs/dpo_qwen3_v2_r4_clean_gpu34
```

输出目录：

- [runs/dpo_qwen3_v2_r4_clean_gpu34](/9950backfile/heruxuan/emocalcu/runs/dpo_qwen3_v2_r4_clean_gpu34)

## 自动评测

### 第一轮历史评测

```bash
scripts/run_model_eval.sh models/Qwen3-8B results/baseline_qwen3_gpu12
scripts/run_model_eval.sh runs/sft_qwen3_lora results/sft_qwen3_gpu12
scripts/run_model_eval.sh runs/dpo_qwen3_baseline_pairs_gpu12 results/dpo_qwen3_gpu12
```

### 第二轮最终评测

当前最终结论使用的是 `quick_eval_q25` 三方同口径对比：

```bash
python3 evaluation/build_comparison_summary.py \
  --results \
  baseline_q25=results/quick_eval_baseline_q25 \
  sft_v2_q25=results/quick_eval_sft_v2_q25 \
  dpo_v2_q25=results/quick_eval_dpo_v2_q25 \
  --output-dir results/comparison_qwen3_v2_final
```

相关目录：

- [results/quick_eval_baseline_q25](/9950backfile/heruxuan/emocalcu/results/quick_eval_baseline_q25)
- [results/quick_eval_sft_v2_q25](/9950backfile/heruxuan/emocalcu/results/quick_eval_sft_v2_q25)
- [results/quick_eval_dpo_v2_q25](/9950backfile/heruxuan/emocalcu/results/quick_eval_dpo_v2_q25)

## Demo 启动

### 推荐：HF 后端加载最终 `DPO v2`

CLI 终端交互：

```bash
CUDA_VISIBLE_DEVICES=1 \
DEMO_MODEL_PATH=/9950backfile/heruxuan/emocalcu/models/Qwen3-8B \
DEMO_ADAPTER_PATH=/9950backfile/heruxuan/emocalcu/runs/dpo_qwen3_v2_r4_clean_gpu34 \
bash scripts/run_demo_cli_hf.sh
```

本地 Gradio：

```bash
CUDA_VISIBLE_DEVICES=1,2 \
DEMO_MODEL_PATH=/9950backfile/heruxuan/emocalcu/models/Qwen3-8B \
DEMO_ADAPTER_PATH=/9950backfile/heruxuan/emocalcu/runs/dpo_qwen3_v2_r4_clean_gpu34 \
bash scripts/run_demo_hf_gpu12.sh
```

说明：

- 这是当前最稳妥的“最终模型演示”路线
- 该路线直接加载本地 LoRA adapter

### 已验证：vLLM + Gradio 基座模型路线

先启动 vLLM：

```bash
CUDA_VISIBLE_DEVICES=5 \
TENSOR_PARALLEL_SIZE=1 \
GPU_MEMORY_UTILIZATION=0.75 \
MAX_MODEL_LEN=8192 \
ENFORCE_EAGER=1 \
PORT=8001 \
deployment/serve_vllm.sh models/Qwen3-8B
```

再启动 Gradio：

```bash
VLLM_PORT=8001 \
CUDA_VISIBLE_DEVICES=1,2 \
bash scripts/run_demo_vllm_gpu12.sh
```

说明：

- 该路线已在本机验证可启动
- 当前验证对象是基座模型 `Qwen3-8B`
- 本仓库没有把 `DPO v2` adapter 作为 `vLLM` 服务链路的已验证默认路径

### Web fallback

```bash
CUDA_VISIBLE_DEVICES=1,2 bash scripts/run_demo_web_gpu12.sh
```

默认端口：`7861`

## 报告与状态文档

- 最终报告：[reports/final_report.md](/9950backfile/heruxuan/emocalcu/reports/final_report.md)
- 最终交付说明：[docs/final_delivery.md](/9950backfile/heruxuan/emocalcu/docs/final_delivery.md)
- 发布打包指南：[docs/publish_guide.md](/9950backfile/heruxuan/emocalcu/docs/publish_guide.md)
- 项目状态：[PROJECT_STATUS.md](/9950backfile/heruxuan/emocalcu/PROJECT_STATUS.md)
- 实验记录：[EXPERIMENT_LOG.md](/9950backfile/heruxuan/emocalcu/EXPERIMENT_LOG.md)

## 打包与上传准备

用于 GitHub 源码打包：

```bash
bash scripts/package_github_source.sh
```

用于 Hugging Face 模型包打包：

```bash
bash scripts/package_hf_model_bundle.sh
```

用于离线整包打包：

```bash
bash scripts/package_offline_bundle.sh
```

## 已知问题

- `MT-Bench/MMLU` 尚未实跑，因此“通用能力不显著下降”仍缺独立 benchmark 补证。
- 当前最强结论基于 `quick_eval_q25` 与代理评分，更适合课程项目复现展示，不等同于大规模人工评测或真实临床验证。
- `DPO v2` 的 `distinct-1/2` 仍低于 baseline，说明表达多样性没有同步提升。
- `avg_exploration` 没有随着 `SFT v2 / DPO v2` 一起明显提升。
- `TRL` 的 `SFTTrainer/DPOTrainer` 在本环境中导入卡住，训练实现已降级为 `Transformers + PEFT + 自定义 DPO loss`。
- `vLLM + Gradio` 虽已验证，但当前使用的是单卡 `--enforce-eager --max-model-len 8192` 降级部署参数。

## 复现说明

- 所有长任务建议在 `tmux` 中执行。
- 所有实验产物统一落在 `results/`、`runs/`、`logs/`。
- 若要快速了解最终状态，建议依次查看：
  - [docs/final_delivery.md](/9950backfile/heruxuan/emocalcu/docs/final_delivery.md)
  - [reports/final_report.md](/9950backfile/heruxuan/emocalcu/reports/final_report.md)
  - [results/comparison_qwen3_v2_final/comparison_summary.md](/9950backfile/heruxuan/emocalcu/results/comparison_qwen3_v2_final/comparison_summary.md)
