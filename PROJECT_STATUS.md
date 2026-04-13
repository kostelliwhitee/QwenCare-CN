# PROJECT STATUS

## 项目概况

- 项目名称：QwenCare-CN
- 当前阶段：Phase 9 最终收尾已完成，当前进入可交付状态
- 当前日期：2026-04-03
- 项目负责人：Codex autonomous agent
- 当前默认最终模型：`runs/dpo_qwen3_v2_r4_clean_gpu34`

## 里程碑状态

| Phase | 名称 | 状态 | 说明 |
| --- | --- | --- | --- |
| 0 | 环境与项目审计 | 已完成 | 已确认 conda、依赖现状与 A800 GPU 可用性 |
| 1 | 数据收集与处理 | 已完成 | 已完成 SmileChat 数据处理与 v2 数据重建 |
| 2 | Baseline 推理与样例验证 | 已完成 | 已完成 `Qwen3-8B` baseline 推理、代理指标与场景评分 |
| 3 | SFT 训练与评测 | 已完成 | 第一轮失败，第二轮 `SFT v2` 已在 `quick_eval_q25` 上超过 baseline |
| 4 | 偏好数据构建 | 已完成 | 第二轮 `DPO v2` 已构建 `303` 条有效 pair |
| 5 | DPO 训练与评测 | 已完成 | `DPO v2` 已完成训练并在当前主评测口径下优于 baseline 与 `SFT v2` |
| 6 | 最终模型选择 | 已完成 | 当前默认最终模型更新为 `DPO v2` |
| 7 | 部署与 demo | 已完成 | 已验证 `vLLM + Gradio` 基座模型路线，并保留 HF/CLI 与 Web fallback |
| 8 | 图表、报告、README、整理 | 已完成 | README、报告、结果摘要已同步到最终状态 |
| 9 | 最终自检与交付清单 | 已完成 | 最终交付说明已整理 |

## 当前已完成

1. 完成 SmileChat 数据处理，生成第一轮 `37227` 条样本并切分。
2. 完成 baseline、第一轮正式 SFT、第一轮正式 DPO。
3. 识别出第一轮失败的根因：`<think>` 模板污染、全序列 loss、数据风格不稳。
4. 实现 `Qwen3 no-think serializer` 与 assistant-only SFT loss。
5. 完成多轮 `v2` 数据重建与审计，最终得到 `v2_r4_clean`。
6. 在 `GPU 4` 上完成正式 `SFT v2`。
7. 在 `GPU 3,4` 上完成正式 `DPO v2`。
8. 完成 `baseline / SFT v2 / DPO v2` 的同口径 `quick_eval_q25` 最终对比。
9. 更新最终结论：当前默认最优模型为 `DPO v2`。
10. 完成 README、最终报告与最终交付说明。

## 当前主结果

### 第一轮正式实验

| model | distinct_1 | distinct_2 | avg_length | avg_overall |
| --- | --- | --- | --- | --- |
| baseline | 0.3333 | 0.7071 | 151.2 | 2.0 |
| sft_v1 | 0.5198 | 0.8056 | 65.8 | 1.0 |
| dpo_v1 | 0.5909 | 0.9474 | 7.3333 | 1.0 |

### 当前最终对比：`quick_eval_q25`

| model | junk_rate | avg_supportiveness | avg_safety | avg_overall |
| --- | --- | --- | --- | --- |
| baseline_q25 | 0.16 | 2.08 | 4.56 | 2.76 |
| sft_v2_q25 | 0.08 | 2.20 | 4.84 | 3.12 |
| dpo_v2_q25 | 0.04 | 2.48 | 4.92 | 3.44 |

完整对比表：

- [results/comparison_qwen3/comparison_summary.md](/9950backfile/heruxuan/emocalcu/results/comparison_qwen3/comparison_summary.md)
- [results/comparison_qwen3_v2_final/comparison_summary.md](/9950backfile/heruxuan/emocalcu/results/comparison_qwen3_v2_final/comparison_summary.md)

## 当前结论

- 第一轮正式 `SFT/DPO` 没有超过 baseline。
- 第二轮 `SFT v2` 已经在当前主评测口径下超过 baseline。
- 第二轮 `DPO v2` 进一步优于 `SFT v2` 与 baseline。
- 当前默认最终模型：
  - `runs/dpo_qwen3_v2_r4_clean_gpu34`
- 当前推荐 demo 路线：
  - 使用 `HF backend + DPO v2 adapter`

## 风险与边界

1. 当前结论主要基于 `quick_eval_q25` 与代理评分，不等同于更大规模人工测评。
2. `MT-Bench/MMLU` 尚未实跑，因此“通用能力未显著下降”仍缺独立 benchmark 证据。
3. `DPO v2` 的 `distinct-1/2` 低于 baseline，多样性没有同步提升。
4. `avg_exploration` 没有明显提高，模型仍更偏“支持性回复”而不是“探索式引导”。
5. 当前 `vLLM + Gradio` 的已验证路线服务对象是基座模型，不是最终 `DPO v2` adapter。

## 交付文件

- README：[README.md](/9950backfile/heruxuan/emocalcu/README.md)
- 最终报告：[reports/final_report.md](/9950backfile/heruxuan/emocalcu/reports/final_report.md)
- 最终交付说明：[docs/final_delivery.md](/9950backfile/heruxuan/emocalcu/docs/final_delivery.md)
- 实验记录：[EXPERIMENT_LOG.md](/9950backfile/heruxuan/emocalcu/EXPERIMENT_LOG.md)
