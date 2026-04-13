# QwenCare-CN 发布与打包指南

## 1. GitHub 源码包

源码仓库建议只上传：

- 代码
- 配置
- 文档
- 轻量结果摘要

不要把以下目录直接放进 GitHub 源码仓库：

- `models/`
- `runs/`
- `logs/`
- 大体积 `data/processed/`

生成 GitHub 源码包：

```bash
bash scripts/package_github_source.sh
```

输出：

- `dist/github_source_bundle/`
- `dist/emocalcu_github_source.tar.gz`

## 2. Hugging Face 模型包

当前推荐上传的不是完整基座，而是最终 LoRA adapter：

- `runs/dpo_qwen3_v2_r4_clean_gpu34`

生成 Hugging Face 模型包：

```bash
bash scripts/package_hf_model_bundle.sh
```

输出：

- `dist/hf_dpo_v2_model/`
- `dist/emocalcu_dpo_v2_hf_model.tar.gz`

该目录内已包含：

- adapter 权重
- tokenizer 文件
- 简要 model card
- `.gitattributes`
- 最终 quick-eval 摘要

## 3. 离线整包

如果希望给他人直接打包下载“项目代码 + 最终模型 adapter”，可使用：

```bash
bash scripts/package_offline_bundle.sh
```

如果还想把本地 `Qwen3-8B` 基座快照一起打进去：

```bash
INCLUDE_BASE_MODEL=1 bash scripts/package_offline_bundle.sh
```

输出：

- `dist/offline_bundle/`
- `dist/emocalcu_offline_bundle.tar.gz`

## 4. 发布前检查

- 确认 `README.md`、`reports/final_report.md`、`docs/final_delivery.md` 一致
- 确认默认最终模型路径为：
  - `runs/dpo_qwen3_v2_r4_clean_gpu34`
- 确认不上传无用的 debug / smoke / 占卡脚本和日志
- 如需公开发布，请补充许可证与数据许可说明
