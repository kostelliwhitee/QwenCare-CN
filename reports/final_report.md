# QwenCare-CN

## 1. 选题背景

心理支持场景要求模型在回复中体现理解、接纳、支持与安全边界。相比一般聊天任务，这类任务更强调回复风格的稳定性，以及对高风险表达的谨慎处理。本项目围绕“中文 AI 心理医生 / 共情对话模型”主题，尝试构建一条可复现的课程项目流水线，并真实执行数据处理、微调、对齐、评测和 demo 交付。

## 2. 研究目标

本项目目标是基于 `Qwen/Qwen3-8B` 构建一个面向轻度情绪困扰场景的中文共情对话系统，并尽量完成：

- 数据准备
- SFT 微调
- DPO 对齐
- 自动评测
- 本地 demo 部署
- 图表、文档与实验记录整理

项目主线目标是：构建一个能够理解用户情绪并提供温暖、共情回复的 AI 心理医生对话模型，提升模型在安慰、鼓励、支持性行为上的表现，同时尽量不显著损失通用能力。

## 3. 方法设计

### 3.1 总体路线

- 基座模型：`Qwen/Qwen3-8B`
- 数据源：`SmileChat`
- 第一轮：LoRA SFT + 小规模 DPO
- 第二轮：`no-think + assistant-only loss` 的 `SFT v2`，再接 `DPO v2`
- 评测：回复质量指标 + 场景模拟评分 + 共情行为代理指标
- 部署：验证 `vLLM + Gradio` 基座模型路线，同时保留 `Transformers` 本地后端与 CLI/Web fallback

### 3.2 与推荐路线的差异

原始目标名为 `Qwen3-8B-Instruct`，但本次环境中可稳定获取与使用的公开仓库是 `Qwen/Qwen3-8B`。此外，`TRL` 中 `SFTTrainer/DPOTrainer` 在当前环境导入卡住，因此训练实现降级为：

- `Transformers + PEFT + Trainer` 完成 SFT
- 本地自定义 pairwise preference loss 完成 DPO

这属于工程兼容性降级，而不是省略训练阶段。

### 3.3 第二轮优化思路

第一轮实验结束后，我们观察到一个关键问题：训练 loss 虽下降，但开放生成质量退化，模型输出中出现了 `assistant`、`recovered`、模板残片等异常 token，说明训练目标与目标场景不一致。

因此第二轮重点做了三件事：

1. 改用 `Qwen3 no-think serializer`，避免训练样本把 `<think>` 模板带进监督目标。
2. 改成 assistant-only loss，只监督最终可见回复，prompt token 全部 mask 为 `-100`。
3. 重建高质量训练数据，从 `SmileChat` 中筛选较稳的样本，并用 baseline 重写中等质量回复，得到 `v2_r4_clean`。

## 4. 数据集与预处理

### 4.1 数据源选择

本项目实际测试过两类公开数据：

1. `thu-coai/esconv`
2. `qiuhuachuan/smile`

最终采用 `SmileChat`，原因如下：

- `thu-coai/esconv` 当前实际使用版本以英文对话为主，不符合“中文共情对话”任务定义
- `SmileChat` 为公开中文心理支持多轮对话，更贴近目标场景

### 4.2 第一轮数据预处理

执行脚本：[prepare_smile_data.py](/9950backfile/heruxuan/emocalcu/scripts/prepare_smile_data.py)

处理流程：

- 读取 `client/counselor` 多轮对话
- 将咨询师回复展开为监督样本
- 统一转为 Alpaca 风格字段：`system / instruction / input / output`
- 基础中文检测
- 基本安全过滤
- 去重
- 按 `8:1:1` 划分训练、验证、测试集

结果：

- 原始文件数：`6796`
- 生成样本数：`37227`
- 过滤 turn 数：`632`
- train：`29781`
- validation：`3723`
- test：`3723`

### 4.3 第二轮数据重建

第二轮没有直接复用原始 `29781` 条训练样本，而是经过多轮重建与审计：

- `v2_r2`：规模恢复，但仍含较多风格漂移和说教表达
- `v2_r3`：显著清理旧污染，但 `gold` 样本仍有较高比例命令式/说教式表达
- `v2_r4`：进一步收紧过滤，训练集高度依赖 `baseline rewrite`
- `v2_r4_clean`：对 `v2_r4` 做轻量后处理，去掉 `<tool_response>` 与 `咨询师：` 前缀污染

最终进入正式 `SFT v2` 的数据集为：

- [train.json](/9950backfile/heruxuan/emocalcu/data/processed/v2_r4_clean/train.json)
- [validation.json](/9950backfile/heruxuan/emocalcu/data/processed/v2_r4_clean/validation.json)

关键统计：

- `num_train_records = 8144`
- `num_validation_records = 1500`
- `num_rewritten_records = 9554`

轻量清洗后的审计结果见 [v2_r4_clean_audit_report.json](/9950backfile/heruxuan/emocalcu/results/v2_r4_clean_audit_report.json)：

- `<tool_response> = 0`
- `consultant_prefix = 0`
- `imperative_hard = 0`
- `medicalized = 0`

## 5. 训练配置

### 5.1 硬件与环境

- GPU：`NVIDIA A800-SXM4-80GB x 8`
- Python 环境：`/9950backfile/heruxuan/path/miniconda3/envs/path`
- `SFT v2` 实际训练：`GPU 4`
- `DPO v2` 实际训练：`GPU 3,4`

### 5.2 关键脚本

- 第一轮 SFT：[run_sft.py](/9950backfile/heruxuan/emocalcu/training/run_sft.py)
- 第一轮 DPO：[run_dpo.py](/9950backfile/heruxuan/emocalcu/training/run_dpo.py)
- 第二轮数据重建：[prepare_sft_v2_data.py](/9950backfile/heruxuan/emocalcu/training/prepare_sft_v2_data.py)
- 第二轮质量规则：[quality_rules.py](/9950backfile/heruxuan/emocalcu/training/quality_rules.py)
- 第二轮 no-think 生成：[qwen3_no_think.py](/9950backfile/heruxuan/emocalcu/training/qwen3_no_think.py)

### 5.3 第二轮正式 SFT v2 配置

正式 `SFT v2` 命令：

```bash
CUDA_VISIBLE_DEVICES=4 bash scripts/run_sft_v2_r4_clean_gpu4.sh
```

核心参数：

- `max_seq_length = 1536`
- `learning_rate = 5e-5`
- `num_train_epochs = 1.0`
- `warmup_ratio = 0.05`
- `lora_r = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`

### 5.4 第二轮正式 DPO v2 配置

正式 `DPO v2` 命令：

```bash
CUDA_VISIBLE_DEVICES=3,4 bash scripts/run_dpo_v2_gpu56.sh \
  data/processed/v2_r4_clean_dpo/dpo_pairs_v2.json \
  runs/sft_qwen3_v2_r4_clean_gpu4 \
  runs/dpo_qwen3_v2_r4_clean_gpu34
```

核心参数：

- `beta = 0.02`
- `learning_rate = 2e-6`
- `num_train_epochs = 1`
- `gradient_accumulation_steps = 8`

## 6. SFT 实验

### 6.1 第一轮正式 SFT

第一轮正式 SFT 在 `GPU 5,6` 上完成：

- 总步数：`1862`
- 最终验证损失：`eval_loss = 1.75098`
- 产物目录：`runs/sft_qwen3_lora`

自动评测目录：`results/sft_qwen3_gpu12`

- `distinct_1 = 0.5198`
- `distinct_2 = 0.8056`
- `avg_length = 65.8`
- `avg_overall = 1.0`

观测：

- 训练收敛，但开放生成出现 `assistant`、`recovered` 等异常 token
- 说明 loss 下降没有转化成目标场景表现提升

### 6.2 第二轮正式 SFT v2

正式 `SFT v2` 产物目录：

- [runs/sft_qwen3_v2_r4_clean_gpu4](/9950backfile/heruxuan/emocalcu/runs/sft_qwen3_v2_r4_clean_gpu4)

训练结果：

- `eval_loss = 0.272341251373291`
- `epoch = 1.0`

在 `quick_eval_q25` 中，`SFT v2` 的结果为：

- `distinct_1 = 0.1160`
- `distinct_2 = 0.4312`
- `avg_length = 195.16`
- `junk_rate = 0.08`
- `avg_empathy = 2.12`
- `avg_supportiveness = 2.20`
- `avg_safety = 4.84`
- `avg_overall = 3.12`

与 baseline 的比较：

- `avg_overall`：`2.76 -> 3.12`
- `junk_rate`：`0.16 -> 0.08`
- `avg_safety`：`4.56 -> 4.84`

这说明第二轮 SFT 已经成功把主线从“低于 baseline”拉回到了“高于 baseline”。

## 7. DPO 实验

### 7.1 第一轮正式 DPO

第一轮偏好数据主要来自 baseline 生成候选，pair 数量为 `354`，正式训练产物目录：

- `runs/dpo_qwen3_baseline_pairs_gpu12`

训练摘要：

- 训练步数：`135`
- 最终 `train_loss ≈ 0.5792`

自动评测：

- `avg_overall = 1.0`
- 未优于 baseline，也未修复异常生成问题

### 7.2 第二轮 DPO v2 数据构造

第二轮不再沿用第一轮的弱 pair，而是基于已经变好的 `SFT v2` 继续构建偏好数据。

数据构建输入：

- seed prompts：`data/processed/dpo_seed_prompts.json`
- baseline candidates：`dpo_candidates_baseline_v2.json`
- `SFT v2` candidates：`dpo_candidates_sft_v2.json`

最终偏好对：

- [dpo_pairs_v2.json](/9950backfile/heruxuan/emocalcu/data/processed/v2_r4_clean_dpo/dpo_pairs_v2.json)
- `pair 数 = 303`

### 7.3 第二轮正式 DPO v2

正式训练输出目录：

- [runs/dpo_qwen3_v2_r4_clean_gpu34](/9950backfile/heruxuan/emocalcu/runs/dpo_qwen3_v2_r4_clean_gpu34)

训练摘要：

- `19` steps
- `train_loss ≈ 0.6907`
- `train_runtime ≈ 161.9s`

在 `quick_eval_q25` 中，`DPO v2` 的结果为：

- `distinct_1 = 0.1087`
- `distinct_2 = 0.4314`
- `avg_length = 197.64`
- `junk_rate = 0.04`
- `avg_empathy = 2.08`
- `avg_supportiveness = 2.48`
- `avg_safety = 4.92`
- `avg_overall = 3.44`

与 `SFT v2` 的比较：

- `avg_overall`：`3.12 -> 3.44`
- `avg_supportiveness`：`2.20 -> 2.48`
- `avg_safety`：`4.84 -> 4.92`
- `junk_rate`：`0.08 -> 0.04`

这说明 `DPO v2` 在当前评测口径下进一步放大了支持性与安全性上的正向收益。

## 8. 评测方案

### 8.1 已实现评测

- 回复质量指标：
  - `distinct-1`
  - `distinct-2`
  - 平均回复长度
  - `negative_behavior_frequency`
  - `junk_rate`
  - `think_leak_rate`
  - `imperative_rate`
  - `too_short_rate`
- 共情行为代理指标：
  - comfort
  - understanding
  - encouragement
  - exploration
- 场景模拟评分：
  - `empathy`
  - `supportiveness`
  - `exploration`
  - `safety`
  - `overall`

关键脚本：

- [run_batch_inference.py](/9950backfile/heruxuan/emocalcu/evaluation/run_batch_inference.py)
- [compute_response_metrics.py](/9950backfile/heruxuan/emocalcu/evaluation/compute_response_metrics.py)
- [score_simulated_dialogues.py](/9950backfile/heruxuan/emocalcu/evaluation/score_simulated_dialogues.py)
- [build_comparison_summary.py](/9950backfile/heruxuan/emocalcu/evaluation/build_comparison_summary.py)

### 8.2 当前主评测口径

当前用于最终模型比较的主口径是 `quick_eval_q25`。它基于项目内部扩展评测集抽取 `25` 条样本，在完全相同的生成和评分脚本下比较：

- baseline
- `SFT v2`
- `DPO v2`

### 8.3 降级说明

原计划中的 `MT-Bench`、`MMLU` 在本轮未完成实跑，当前只保留了可重复执行的小规模自动评测与场景评分。这属于课程项目中的中度降级，意味着我们能够较好比较“共情支持主线”上的相对改进，但还不能用独立 benchmark 证明通用能力未显著下降。

## 9. 实验结果

### 9.1 第一轮正式实验结果

| model | distinct_1 | distinct_2 | avg_length | avg_overall |
| --- | --- | --- | --- | --- |
| baseline | 0.3333 | 0.7071 | 151.2 | 2.0 |
| sft_v1 | 0.5198 | 0.8056 | 65.8 | 1.0 |
| dpo_v1 | 0.5909 | 0.9474 | 7.3333 | 1.0 |

### 9.2 第二轮最终结果：`quick_eval_q25`

| model | distinct_1 | distinct_2 | avg_length | junk_rate | avg_empathy | avg_supportiveness | avg_safety | avg_overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_q25 | 0.1283 | 0.4721 | 160.28 | 0.16 | 1.96 | 2.08 | 4.56 | 2.76 |
| sft_v2_q25 | 0.1160 | 0.4312 | 195.16 | 0.08 | 2.12 | 2.20 | 4.84 | 3.12 |
| dpo_v2_q25 | 0.1087 | 0.4314 | 197.64 | 0.04 | 2.08 | 2.48 | 4.92 | 3.44 |

结果目录：

- [results/comparison_qwen3_v2_final/comparison_summary.md](/9950backfile/heruxuan/emocalcu/results/comparison_qwen3_v2_final/comparison_summary.md)

## 10. 对比分析

### 10.1 观测结果

- 第一轮 `SFT/DPO` 明显没有达到目标，表现低于 baseline。
- 第二轮 `SFT v2` 说明“no-think + assistant-only loss + 更干净的数据”确实解决了主线问题。
- 第二轮 `DPO v2` 又在 `SFT v2` 基础上进一步提升了：
  - `avg_overall`
  - `avg_supportiveness`
  - `avg_safety`
  - `junk_rate`

### 10.2 推测解释

- 第一轮失败的主要原因，不是“微调方向错误”，而是训练目标与数据格式把模型训偏了。
- 第二轮的收益主要来自：
  - 去掉 `<think>` 污染
  - 只监督 assistant 回复
  - 更严格的数据审计与 baseline rewrite
- `DPO v2` 的提升更集中在“更稳、更支持、更少垃圾回复”，而不是“更多样性”或“更强探索性”。

### 10.3 当前最终模型选择

当前默认最终模型：

- [runs/dpo_qwen3_v2_r4_clean_gpu34](/9950backfile/heruxuan/emocalcu/runs/dpo_qwen3_v2_r4_clean_gpu34)

选择依据：

- 在当前项目内的同口径 `quick_eval_q25` 中，`DPO v2` 是表现最好的模型
- 它优于 baseline，也优于 `SFT v2`
- 但这一选择仍然带有边界：尚未由 `MT-Bench/MMLU` 做通用能力补证

## 11. 失败案例与局限性

### 11.1 第一轮失败案例

- `SFT v1` 输出异常：
  - `assistant`
  - `recovered`
  - `assistant\noplayer\nassistant`
- `DPO v1` 输出异常：
  - `assistant`
  - 空串
  - `ypse`

### 11.2 当前局限性

- 当前未完成 `MT-Bench`、`MMLU` 等更完整的通用能力 benchmark
- 当前结论主要依赖代理指标与小规模场景评分
- `DPO v2` 的 `distinct-1/2` 低于 baseline，提示表达多样性存在一定退化
- `avg_exploration` 没有明显提升，说明“引导继续表达”的能力仍有改进空间
- `v2_r4_clean` 训练集规模仅 `8144`，且较依赖 `baseline rewrite`，存在风格同质化风险
- 当前结论不适用于医疗诊断或真实治疗场景

## 12. 部署与 demo

### 12.1 已验证部署路线

- `vLLM + Gradio`
  - 已在基座模型 `Qwen3-8B` 上验证可启动
  - 当前使用单卡 `--enforce-eager --max-model-len 8192` 的降级参数
- `FastAPI + HTML`
  - 已验证可启动，作为 Web fallback
- `Transformers + CLI`
  - 已可直接加载本地 model / adapter 做终端交互

### 12.2 当前推荐 demo 路线

如果目标是演示“当前最终模型”，推荐直接用 `HF backend + DPO v2 adapter`：

```bash
CUDA_VISIBLE_DEVICES=1 \
DEMO_MODEL_PATH=/9950backfile/heruxuan/emocalcu/models/Qwen3-8B \
DEMO_ADAPTER_PATH=/9950backfile/heruxuan/emocalcu/runs/dpo_qwen3_v2_r4_clean_gpu34 \
bash scripts/run_demo_cli_hf.sh
```

或：

```bash
CUDA_VISIBLE_DEVICES=1,2 \
DEMO_MODEL_PATH=/9950backfile/heruxuan/emocalcu/models/Qwen3-8B \
DEMO_ADAPTER_PATH=/9950backfile/heruxuan/emocalcu/runs/dpo_qwen3_v2_r4_clean_gpu34 \
bash scripts/run_demo_hf_gpu12.sh
```

### 12.3 安全边界

- demo 中已加入高风险表达的基础安全应答逻辑
- 系统声明仅用于课程项目与情绪支持研究演示
- 不构成专业医疗诊断或治疗建议

## 13. 未来工作

- 补跑 `MT-Bench/MMLU` 子集，验证通用能力是否未显著下降
- 扩展更大规模的 held-out 场景评测与人工评审
- 引入更强的本地 judge，提高 `DPO` pair 质量与规模
- 针对“探索性”单独设计数据与偏好项，提升继续表达引导能力
- 研究如何把当前 `DPO v2` adapter 更平滑地接入 `vLLM` 服务链路

## 14. 参考文献

1. Qwen Team. `Qwen/Qwen3-8B`. Hugging Face model repository.
2. Qiu Huachuan et al. `Smile`. GitHub repository for Chinese psychological support dialogues.
3. THU-COAI. `ESConv`. Hugging Face dataset repository.
4. Hugging Face. `PEFT` and `Transformers` documentation.

## 15. 附录

### 15.1 关键结果文件

- 最终对比表：
  - [results/comparison_qwen3_v2_final/comparison_summary.md](/9950backfile/heruxuan/emocalcu/results/comparison_qwen3_v2_final/comparison_summary.md)
- 当前最终模型：
  - [runs/dpo_qwen3_v2_r4_clean_gpu34](/9950backfile/heruxuan/emocalcu/runs/dpo_qwen3_v2_r4_clean_gpu34)
- 项目状态：
  - [PROJECT_STATUS.md](/9950backfile/heruxuan/emocalcu/PROJECT_STATUS.md)
- 实验日志：
  - [EXPERIMENT_LOG.md](/9950backfile/heruxuan/emocalcu/EXPERIMENT_LOG.md)

### 15.2 关键命令

正式 `SFT v2`：

```bash
CUDA_VISIBLE_DEVICES=4 bash scripts/run_sft_v2_r4_clean_gpu4.sh
```

正式 `DPO v2`：

```bash
CUDA_VISIBLE_DEVICES=3,4 bash scripts/run_dpo_v2_gpu56.sh \
  data/processed/v2_r4_clean_dpo/dpo_pairs_v2.json \
  runs/sft_qwen3_v2_r4_clean_gpu4 \
  runs/dpo_qwen3_v2_r4_clean_gpu34
```

最终模型 CLI demo：

```bash
CUDA_VISIBLE_DEVICES=1 \
DEMO_MODEL_PATH=/9950backfile/heruxuan/emocalcu/models/Qwen3-8B \
DEMO_ADAPTER_PATH=/9950backfile/heruxuan/emocalcu/runs/dpo_qwen3_v2_r4_clean_gpu34 \
bash scripts/run_demo_cli_hf.sh
```
