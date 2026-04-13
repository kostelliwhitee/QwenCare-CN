# 执行计划

## 任务拆解

1. 审计环境、GPU、conda、现有依赖与模型缓存
2. 搭建仓库骨架、状态管理与复现配置
3. 下载公开中文情绪支持/共情数据集
4. 清洗数据、去重、安全过滤、统一格式为 Alpaca
5. 跑通 baseline 推理，建立对比样例
6. 执行 LoRA SFT 训练
7. 基于模型候选回复构造偏好对
8. 执行 DPO 训练或记录降级原因
9. 跑自动评测与图表生成
10. 部署 vLLM 服务与 Gradio demo
11. 撰写报告与仓库收尾

## 依赖关系

- 数据准备依赖可用网络或本地缓存
- Baseline 依赖基座模型可下载或已缓存
- SFT 依赖数据、`peft`、训练框架
- DPO 依赖 SFT 产物与偏好对脚本
- Demo 依赖最终模型与 `vllm`/`gradio`
- 报告依赖实验日志与结果文件

## 里程碑

| 里程碑 | 验收标准 |
| --- | --- |
| M0 审计完成 | 文档中明确环境、GPU、依赖、风险 |
| M1 数据可用 | `data/processed/train.json` 等文件已生成 |
| M2 Baseline 可跑 | 基座模型样例推理成功并保存 |
| M3 SFT 可复现 | 训练脚本、配置、日志与 checkpoint 可追溯 |
| M4 DPO 可复现 | 偏好数据与训练命令可追溯 |
| M5 评测完成 | 指标表与图表自动生成 |
| M6 Demo 可启动 | vLLM 与 Gradio 本地启动命令可用 |
| M7 文档完备 | README、状态、报告与实验日志齐全 |

## 风险项

1. 下载速度或网络权限影响模型与数据获取
2. `LLaMA-Factory` 与现有 `transformers` 版本可能不兼容
3. vLLM 安装耗时较长或受 CUDA/torch 版本影响
4. 公开中文心理支持数据规模可能不足
5. DPO 对齐收益可能不显著

## 降级策略

### Level 0

- LLaMA-Factory + LoRA SFT + DPO
- MMLU/MT-Bench 子集与共情指标并行
- vLLM + Gradio 完整部署

### Level 1

- SFT 全量，DPO 小规模
- 通用 benchmark 使用子集
- 用户模拟场景缩减但不少于 3 个

### Level 2

- 保证 SFT 完成
- DPO 仅保留小样本验证或替代对齐实验
- 通用 benchmark 保留一个代表性任务

### Level 3

- 保证数据处理、baseline vs SFT、至少一类自动评测、可运行 demo、完整报告
- 若 DPO 未完成，必须写清尝试过程、失败原因与后续建议

## 预期交付物

- `README.md`
- `PROJECT_STATUS.md`
- `EXPERIMENT_LOG.md`
- `configs/`
- `scripts/`
- `training/`
- `evaluation/`
- `deployment/`
- `demo/`
- `results/`
- `reports/final_report.md`
