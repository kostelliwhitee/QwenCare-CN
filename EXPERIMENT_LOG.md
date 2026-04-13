# EXPERIMENT LOG

## 2026-03-19

### Session 1: 项目初始化与环境审计

- 操作：初始化仓库、创建目录、定位 conda、检查 Python/torch/依赖状态
- 关键事实：
  - 工作目录初始为空
  - `python3` 为 `3.10.4`
  - conda 安装位于 `/9950backfile/heruxuan/path/miniconda3`
  - 可用环境包含 `base` 与 `path`
  - `path` 环境 Python 为 `3.10.19`
  - `torch` 版本为 `2.10.0+cu128`
  - `transformers` 存在，`peft` 缺失
  - `datasets`、`deepspeed`、`accelerate` 已安装
  - `vllm`、`gradio`、`llamafactory` 缺失
  - 沙箱内 `torch.cuda.is_available()` 为 `False`
- 影响：
  - 沙箱内不能直接判断真实 GPU 状态
  - 需先完成脱沙箱 GPU 核验与依赖补齐
- 后续动作：
  - 继续检查本地模型缓存
  - 补写环境与审计文档
  - 准备数据处理与训练脚本骨架

### Session 2: GPU 脱沙箱核验

- 操作：在 conda `path` 环境中脱离沙箱运行 `torch.cuda` 检查
- 结果：
  - `torch.cuda.is_available()` 为 `True`
  - `torch.cuda.device_count()` 为 `8`
  - 设备名称均为 `NVIDIA A800-SXM4-80GB`
- 决策：
  - 后续训练、推理与评测默认绑定 `CUDA_VISIBLE_DEVICES=5,6`
  - 可以继续真实推进 SFT / DPO / vLLM 部署路线

### Session 3: 数据源验证与替换

- 操作：
  - 测试下载 `thu-coai/esconv`
  - 检索公开中文心理支持数据源
  - 下载并解压 `qiuhuachuan/smile`
- 关键发现：
  - Hugging Face 上拉取到的 `thu-coai/esconv` 版本是英文情绪支持对话，不符合中文任务定义
  - `qiuhuachuan/smile` 仓库公开提供中文心理健康支持对话 `SmileChat`
- 决策：
  - 以 `SmileChat` 作为当前主训练语料
  - 将 ESConv 记录为被放弃的候选数据源，并在报告中说明原因

### Session 4: SmileChat 转换与质检

- 操作：
  - 编写并执行 `scripts/prepare_smile_data.py`
  - 将多轮对话转换为 Alpaca 风格 SFT 样本
  - 生成 `train/validation/test` 与 `dataset_stats.json`
  - 计算基础代理指标 `results/data_quality_metrics.json`
- 结果：
  - 原始对话文件数：6796
  - 样本数：37227
  - 过滤 turn 数：632
  - 切分结果：29781 / 3723 / 3723
  - 测试集代理指标：
    - `distinct_1 = 0.0063`
    - `distinct_2 = 0.1346`
    - `avg_length = 100.61`
    - `negative_behavior_frequency = 0.0086`
- 观察：
  - 数据整体为中文、多轮、心理支持场景，适合作为 SFT 主语料
  - 部分回复存在较强建议导向，后续需依赖过滤与 DPO 进一步优化

### Session 5: 依赖补齐与环境隔离

- 操作：
  - 将 `peft`、`trl`、`gradio`、`evaluate`、`jieba`、`matplotlib`、`seaborn`、`scikit-learn`、`rouge_chinese` 安装到仓库本地 `.vendor/`
  - 修复 `.vendor` 对 `conda` 入口造成的 `PYTHONPATH` 污染问题
  - 将统一入口改为直接使用目标 env 的 `bin/python`
- 结果：
  - `peft 0.18.1`
  - `trl 0.29.0`
  - `gradio 6.9.0`
  - 通过 `scripts/run_in_conda.sh` 已可正常 import 关键依赖

### Session 6: 基座模型仓库名纠正

- 操作：
  - 查询 Hugging Face 模型索引
- 发现：
  - 公开可见仓库名为 `Qwen/Qwen3-8B`
  - 初始尝试的 `Qwen/Qwen3-8B-Instruct` 返回 401 / not found
- 决策：
  - 当前 baseline 改为下载并使用 `Qwen/Qwen3-8B`
  - 在文档中注明用户目标名与公开仓库名之间的差异

### Session 7: 训练框架兼容性降级

- 操作：
  - 测试 `trl` 中 `SFTTrainer` 和 `DPOTrainer` 的导入可用性
- 结果：
  - `trl` 基础模块可导入
  - `trl.trainer.sft_trainer` 与 `trl.trainer.dpo_trainer` 在当前环境中长时间无响应
- 决策：
  - SFT 改用 `Transformers + PEFT + Trainer`
  - DPO 改用本地自定义 pairwise preference loss 实现
  - 记录为兼容性降级，不再依赖 `TRL trainer` 主路径

### Session 8: tmux 执行约束与主模型下载迁移

- 用户新要求：
  - 所有长任务运行时必须在 `tmux` 中执行
- 操作：
  - 检查 `tmux` 和 GPU 实时状态
  - 发现 GPU 4-7 算力空闲，但存在低显存保留进程
  - 新建 `tmux` 会话 `emocalcu_qwen3_download`
  - 将 `Qwen/Qwen3-8B` 下载切换到带重试 wrapper 的 `tmux` 会话中
- 当前状态：
  - `Qwen3-8B` 尚未完整下载
  - 已确认 `model-00005-of-00005.safetensors` 落盘
  - 下载缓存继续增长，已达到约 `8.0G`

### Session 9: smoke 模型完成与 smoke SFT 启动

- 操作：
  - 确认 `Qwen/Qwen2.5-0.5B-Instruct` 下载完成
  - 新建 `tmux` 会话 `emocalcu_smoke_sft`
  - 使用 `GPU 4`、`128/32` 子集启动 smoke SFT
- 结果：
  - 已在日志中看到模型权重加载与数据集 `map` 完成
  - 说明自定义 SFT 脚本至少已经进入真实初始化阶段

### Session 10: baseline 推理报错修复与首轮结果

- 操作：
  - 在 `tmux` 会话 `emocalcu_baseline_gpu12` 中使用 `GPU 1,2` 运行 baseline
  - 修复 `evaluation/run_batch_inference.py` 中将 `BatchEncoding` 当作 tensor 使用导致的 `AttributeError`
  - 同步修复 `training/infer_hf.py` 与 `training/generate_dpo_candidates.py` 的同类调用方式
- 原始报错：
  - `AttributeError` at `inputs_tensor.shape[0]`
- 决策：
  - 统一改为 `return_dict=True` 后用 `inputs["input_ids"].shape[-1]` 获取 prompt 长度
  - 后续所有推理入口复用同一模式，避免再次触发

### Session 11: `<think>` 泄漏与前缀噪声清洗

- 操作：
  - baseline 首轮结果中观察到 `<think>` 泄漏和 `._.` 前缀噪声
  - 在 `evaluation/run_batch_inference.py`、`training/infer_hf.py`、`training/generate_dpo_candidates.py` 中加入：
    - 更强 system prompt
    - `bad_words_ids` 屏蔽 `<think>` / `</think>`
    - 回退重生成逻辑
    - 首行噪声剥离逻辑
  - 在 `GPU 1,2` 上于 `tmux` 中重跑 baseline
- 最新 baseline 结果：
  - 结果目录：`results/baseline_qwen3_gpu12`
  - `distinct_1 = 0.3333`
  - `distinct_2 = 0.7071`
  - `avg_length = 151.2`
  - `negative_behavior_frequency = 0.0`
  - `avg_overall = 2.0`
- 观察：
  - `<think>` 泄漏已被抑制
  - 回复前缀噪声已清除
  - baseline 输出风格整体更适合后续课程展示

### Session 12: smoke SFT 完成

- 操作：
  - 持续观察 `tmux` 会话 `emocalcu_smoke_sft`
- 结果：
  - smoke SFT 成功完成
  - 输出目录：`runs/smoke_sft_qwen25`
  - 训练日志显示：
    - `train_runtime ≈ 37.68s`
    - `train_loss ≈ 2.375`
    - `eval_loss ≈ 2.2155`
- 结论：
  - `Transformers + PEFT + Trainer` 路线可以在当前环境中真实完成训练
  - 可继续切换到 Qwen3 正式 SFT

### Session 13: 正式 Qwen3 SFT 启动与训练稳定性修复

- 操作：
  - 在 `GPU 5,6` 上通过 `tmux` 会话 `emocalcu_sft_gpu56` 启动正式 LoRA SFT
  - 修复 `training/run_sft.py` 中 collator 对 `labels` 直接调用 `tokenizer.pad` 导致的嵌套张量报错
  - 改为手动 padding：
    - `input_ids -> pad_token_id`
    - `attention_mask -> 0`
    - `labels -> -100`
- 原始报错：
  - `ValueError: features ('labels') have excessive nesting`
- 当前训练观测：
  - 训练总步数：`1862`
  - 已完成中间评估：`eval_loss = 1.885`
  - 训练日志继续前进至 `step 130+`
  - 最近 loss 大致降至 `1.875 ~ 1.891`
- 风险：
  - 日志提示当前内核版本 `5.4.0` 低于推荐值，理论上存在 hang 风险
  - 当前实际训练持续推进，暂未观测到中断

### Session 14: SFT 后评测与 DPO 数据构造脚本补齐

- 操作：
  - 新增 `scripts/run_model_eval.sh`
  - 新增 `scripts/run_sft_eval_gpu12.sh`
  - 新增 `scripts/build_dpo_data.sh`
  - 新增 `scripts/build_dpo_data_gpu12.sh`
  - 推理与候选生成脚本增加 LoRA adapter 自动识别加载逻辑
- 目的：
  - 让 `runs/sft_qwen3_lora` 在训练完成后可直接进行评测
  - 让 DPO 候选采样与 pair 构造可以直接基于 SFT adapter 运行
- 当前结论：
  - 主线已从“训练脚本能跑”推进到“训练后链路已打通”

### Session 15: 正式 SFT 完成与自动评测结果

- 操作：
  - 持续观察 `tmux` 会话 `emocalcu_sft_gpu56`
  - 等待训练完成并收集最终产物
  - 自动触发 post-SFT 评测流程
- 结果：
  - 最终产物目录：`runs/sft_qwen3_lora`
  - 最终 checkpoint：`checkpoint-1862`
  - 最终验证损失：
    - `eval_loss = 1.75098`
    - `epoch = 2.0`
  - 自动评测目录：`results/sft_qwen3_gpu12`
  - 自动评测指标：
    - `distinct_1 = 0.5198`
    - `distinct_2 = 0.8056`
    - `avg_length = 65.8`
    - `avg_overall = 1.0`
- 关键观察：
  - 虽然训练损失下降，但 SFT 模型在开放生成评测中出现明显异常输出
  - 例子包括：`assistant`、`recovered`、`assistant\noplayer\nassistant`
  - 说明当前 SFT 训练并未稳定学到目标心理支持风格

### Session 16: 首轮 DPO 自动启动失败

- 操作：
  - `tmux` 会话 `emocalcu_wait_dpo_gpu56` 在 `dpo_pairs.json` 生成后自动启动 DPO
  - 初始多卡 DPO 因 `ref_model` 设备放置不一致失败
- 修复：
  - 在 `training/run_dpo.py` 中显式读取 `LOCAL_RANK`
  - 将 `policy_model` 与 `ref_model` 都移动到 `cuda:{local_rank}`
  - 关闭 `use_cache` 以适配训练

### Session 17: baseline 驱动 DPO 数据重建

- 操作：
  - 放弃使用 SFT 输出构造 DPO pair
  - 改为使用 baseline `Qwen3-8B` 生成候选并重新筛选
- 结果：
  - 候选文件：`data/processed/dpo_candidates_baseline.json`
  - 偏好对文件：`data/processed/dpo_pairs_baseline.json`
  - prompt 数：`512`
  - pair 数：`354`
  - 代理有效 pair：`354 / 354`
- 决策：
  - 将此数据作为正式 DPO 输入
  - 记录为“DPO 数据源降级”，但不放弃 DPO 阶段

### Session 30: `v2_r2` 数据审计与严格版 `v2_r3` 重建启动

- 操作：
  - 核对 `data/processed/v2_r2/summary.json`
  - 新增 `training/audit_sft_dataset.py`
  - 对 `data/processed/v2_r2/train.json` 生成审计报告 `results/v2_r2_audit_report.json`
  - 收紧 `training/quality_rules.py` 中的风格漂移与命令式规则
  - 收紧 `training/prepare_sft_v2_data.py` 中的低质量回退保留逻辑
  - 新增 `scripts/build_sft_v2_data_strict_gpu2.sh`
  - 在 `tmux` 会话 `emocalcu_build_sft_v2_gpu2_r3` 中启动严格版 `v2_r3` 数据重建
- 结果：
  - `v2_r2` 规模恢复正常：
    - `train = 14833`
    - `validation = 1500`
    - `rewritten = 8000`
  - 但审计显示训练集仍含明显污染：
    - `style_drift = 2293`
    - `imperative = 986`
    - `source_type::gold = 8127`
    - `source_type::baseline_rewrite = 6706`
  - 典型问题样本包括：
    - `.superstar`
    - `ionale`
    - `喜欢就去追`
    - `帮助你是我的责任和使命`
    - `亲亲 / 妹妹 / 宝宝` 等人设化称呼
- 决策：
  - 当前不启动正式 `SFT v2`
  - 当前不继续 `DPO v2`
  - 先用更严格的风格过滤和更保守的回退逻辑重建 `v2_r3`
  - 只有当 `v2_r3` 抽样审计通过后，才允许进入正式 `SFT v2`

### Session 31: `v2_r3` 正式严格数据重建完成

- 操作：
  - 使用带周期性进度日志的 `training/prepare_sft_v2_data.py`
  - 在全量 `train + validation` 输入上完成正式 `v2_r3` 重建
  - 输出目录：`data/processed/v2_r3`
  - 日志：`logs/build_sft_v2_data_r3.log`
- 结果：
  - `num_input_records = 32321`
  - `num_train_records = 20005`
  - `num_validation_records = 1500`
  - `num_rewritten_records = 8000`
  - `num_gold_records = 13505`
  - `num_rewrite_records = 8000`
  - `dropped_for_low_quality = 10816`
- 关键观察：
  - 严格版过滤显著扩大了训练集规模，相比 `v2_r2` 的 `14833` 条，`v2_r3` 达到 `20005` 条
  - 这轮完成的是“严格数据重建”，不是新的正式 `SFT v2`
  - 当前仓库里最新的全量训练结果仍然是：
    - 第一轮正式 SFT：`runs/sft_qwen3_lora`
    - 第一轮正式 DPO：`runs/dpo_qwen3_baseline_pairs_gpu12`
    - v2 方向仅完成了 smoke 训练：`runs/sft_qwen3_v2_smoke`
- 决策：
  - 当前下一步应是对 `v2_r3` 做抽样审计，再决定是否启动正式 `SFT v2`
  - 不能把 `v2_r3` 完成误写成“已获得新的正式训练结果”

### Session 32: `v2_r3` 抽样审计与 go/no-go 结论

- 操作：
  - 对 `data/processed/v2_r3/train.json` 生成审计报告 `results/v2_r3_audit_report.json`
  - 抽样检查 `80` 条记录，并额外检查 `baseline_rewrite` 与 `gold` 可疑样本
  - 重点检查：
    - `机器人 / 姐妹 / 妹妹 / 亲亲 / 宝宝 / 宝贝 / 哥哥`
    - 英文前缀噪声
    - 命令式/说教式表达
    - 医疗化表达
- 结果：
  - 旧污染已显著下降：
    - `assistant = 0`
    - `recovered = 0`
    - `<think> = 0`
    - `.superstar = 0`
    - `ionale = 0`
  - 但仍存在明显风格风险：
    - 全量计数：
      - `妹妹 = 21`
      - `姐妹 = 7`
      - `亲亲 = 1`
      - `宝宝 = 41`
      - `哥哥 = 21`
    - 规则统计：
      - `gold imperative_hard = 1011 / 13348`
      - `baseline_rewrite imperative_hard = 851 / 6657`
      - `gold persona_terms = 67`
      - `baseline_rewrite persona_terms = 24`
      - `gold medicalized = 55`
      - `baseline_rewrite medicalized = 103`
  - 抽样观察：
    - `baseline_rewrite` 整体风格明显好于第一轮，但仍有少量“亲爱的”“加油”“过度建议化”的样本
    - 更大的问题来自 `gold`：仍有大量“你应该/你需要/首先可以/建议你/一定要”式回答
    - `gold` 中仍有明显课程答疑感、直接指导感和轻度医疗化样本
- 结论：
  - `v2_r3` 明显优于 `v2_r2`
  - 但 **当前不建议直接启动正式 `SFT v2`**
  - 原因不是旧模板污染，而是 `gold` 样本里的说教/命令式表达比例仍然偏高，直接训练仍有较大概率把模型拉回“建议型答疑”风格
- 决策：
  - 当前 `go/no-go` 结论为：`NO-GO`
  - 下一步应继续收紧过滤器，优先处理 `gold` 样本中的命令式和人设化表达，再生成下一版严格训练集后复审

### Session 33: 收紧过滤器并启动 `v2_r4` 候选数据生成

- 操作：
  - 继续收紧 `training/quality_rules.py`
    - 新增 `advice_heavy` 结构化建议检测
    - 扩展 `style_drift`，覆盖 `亲爱的 / hi / 助理 / 祝你好运 / 随时来找我聊天` 等模式
  - 更新 `training/prepare_sft_v2_data.py`
    - 引入 `passes_keep_gate`
    - `gold` 样本必须同时满足“质量分数达标 + 风格门控通过”才能直接保留
    - 原样本回退保留也必须通过同一门控
  - 新增 `scripts/build_sft_v2_data_r4_gpu2.sh`
  - 在 `tmux` 会话 `emocalcu_build_sft_v2_gpu2_r4` 中启动更严格的 `v2_r4` 候选数据生成
- 配置：
  - `gold_threshold = 2.0`
  - `rewrite_threshold = 0.0`
  - `rewrite_margin = 1.0`
  - `rewrite_limit = 10000`
  - `fallback_keep_threshold = 2.0`
  - `progress_every = 250`
- 当前状态：
  - `python training/prepare_sft_v2_data.py ... --output-dir data/processed/v2_r4` 已确认在运行
  - `tee logs/build_sft_v2_data_r4.log` 已确认在运行
- 目的：
  - 继续压低 `gold` 样本中的说教/命令式回答
  - 为正式 `SFT v2` 生成一版更保守的候选训练集

### Session 31: 正式 `v2_r3` 首次重建失败与可观测性修复

- 操作：
  - 检查 `tmux` 会话 `emocalcu_build_sft_v2_gpu2_r3`
  - 检查 `logs/build_sft_v2_data_r3.log`
  - 检查 `data/processed/v2_r3/`
  - 检查 `tmux pane` 子进程与会话状态
  - 为 `training/prepare_sft_v2_data.py` 增加周期性进度日志
- 结果：
  - 首次正式 `v2_r3` 运行只打印到权重加载完成，没有产出任何结果文件
  - 后续确认该 `tmux` 会话已经消失，`data/processed/v2_r3/` 仍为空目录
  - 因此这次运行应判定为失败，不能继续等待
  - 已在 `prepare_sft_v2_data.py` 中加入周期性日志：
    - `processed`
    - `kept`
    - `rewritten`
    - `dropped_for_low_quality`
- 决策：
  - 不再把“日志停在 loading weights”视为自动正常
  - 先做小规模复现，确认严格版数据构建逻辑本身可稳定完成
  - 小规模通过后，再重启正式 `v2_r3`

### Session 32: 小规模 `v2_debug_small` 复现成功

- 操作：
  - 从 `data/processed/train.json` 和 `data/processed/validation.json` 中抽取：
    - `train_128`
    - `validation_32`
  - 在 `tmux` 会话 `emocalcu_v2_debug_gpu2` 中运行严格版数据构建 debug
  - 参数：
    - `rewrite_limit = 16`
    - `validation_size = 16`
    - `dpo_seed_size = 32`
    - `progress_every = 10`
- 结果：
  - `logs/v2_debug_small.log` 中成功输出周期性进度
  - 最终产物：
    - `data/processed/v2_debug_small/summary.json`
  - 关键统计：
    - `num_input_records = 154`
    - `num_train_records = 125`
    - `num_validation_records = 16`
    - `num_rewritten_records = 16`
    - `dropped_for_low_quality = 13`
- 结论：
  - 严格版 `prepare_sft_v2_data.py` 逻辑本身已被验证可工作
  - 当前应重启正式 `v2_r3`，而不是继续等待旧失败会话

### Session 18: 正式 DPO 完成

- 操作：
  - 使用 baseline 构造的 pair 启动正式 DPO
- 结果：
  - 输出目录：`runs/dpo_qwen3_baseline_pairs_gpu12`
  - 训练步数：`135`
  - 训练轮数：`3`
  - 最终 `train_loss ≈ 0.5792`
  - 自动评测目录：`results/dpo_qwen3_gpu12`
- 观察：
  - DPO 未能修复 SFT 中的异常 token 与空回复问题
  - 当前自动评测中 DPO 仍弱于 baseline

### Session 19: 对比汇总与最终模型选择

- 操作：
  - 汇总 baseline / SFT / DPO 三组结果
  - 生成对比表与柱状图
- 结果：
  - 汇总目录：`results/comparison_qwen3`
  - baseline `avg_overall = 2.0`
  - sft `avg_overall = 1.0`
  - dpo `avg_overall = 1.0`
- 结论：
  - 当前默认部署模型选择 `baseline Qwen3-8B`
  - 不宣称 SFT 或 DPO 带来显著提升

### Session 20: demo 部署路径收口

- 操作：
  - 重写 `demo/app.py`
  - 增加 `Transformers` 本地推理后端
  - 保留 `vLLM` OpenAI API 后端
  - 增加高风险表达的基础安全响应
  - 新增启动脚本：
    - `scripts/run_demo_hf_gpu12.sh`
    - `scripts/run_demo_vllm_gpu12.sh`
- 当前状态：
  - 当前环境未检测到 `vllm`
  - 按降级策略，默认交付 `HF + Gradio` 兜底脚本

### Session 21: Gradio 启动诊断

- 操作：
  - 使用 `importtime` 对 `gradio` 导入链路进行诊断
  - 使用本地 `curl` 对 `demo/app.py` 的 `7860` 端口进行探活
- 结果：
  - `gradio 6.9.0`、`huggingface_hub 1.7.1`、`httpx 0.28.1`
  - `gradio` 导入链路在 `httpx / huggingface_hub` 附近表现为极慢初始化
  - `demo/app.py` 在 60 秒窗口内未成功监听本地 `7860` 端口
- 结论：
  - 当前 demo 脚本已具备交付形态
  - 但部署验证尚未完成，需在后续继续排查 `gradio` 启动阻塞

### Session 22: Web fallback 补充与验证

- 操作：
  - 新增 `demo/backend.py`
  - 新增 `demo/web_app.py`
  - 新增 `scripts/run_demo_web_gpu12.sh`
  - 将后端推理改为 lazy load，避免服务启动时先加载模型
- 结果：
  - `python -m uvicorn demo.web_app:app --host 127.0.0.1 --port 7861` 可成功启动
  - 已观测到 `Application startup complete` 与 `Uvicorn running on http://127.0.0.1:7861`
- 决策：
  - 当前课程项目 demo 交付路径调整为 `FastAPI + HTML` fallback
  - `Gradio` 脚本继续保留，作为后续兼容性修复目标

### Session 23: vLLM 安装尝试

- 操作：
  - 用户确认允许继续安装 `vllm`
  - 在 `tmux` 会话 `emocalcu_install_vllm` 中启动首次安装
  - 观察到安装卡在旧版 `huggingface-hub / transformers` 的卸载阶段
- 关键现象：
  - 日志中出现：
    - `Can't uninstall 'huggingface-hub'. No files were found to uninstall.`
    - `Can't uninstall 'transformers'. No files were found to uninstall.`
  - 说明当前 conda 环境中存在不完整 metadata，导致标准 uninstall 流程异常缓慢
- 处理：
  - 终止首轮安装
  - 在 `tmux` 会话 `emocalcu_install_vllm_fix` 中改用 `pip install --ignore-installed --no-input vllm`
- 当前状态：
  - 新安装流程已开始拉取和写入依赖
  - 后续发现安装最终在下载 `sentry_sdk` wheel 时失败

### Session 24: vLLM 安装真实阻塞确认

- 操作：
  - 继续检查 `logs/install_vllm_fix.log`
  - 复核 `tmux` 会话与 `import vllm`
- 结果：
  - 当前 `tmux` 中不再有项目相关活跃安装会话
  - `import vllm` 仍失败
  - 日志末尾明确报错：
    - `ProxyError('Cannot connect to proxy.')`
    - 失败对象为 `sentry_sdk` wheel 下载
- 结论：
  - 当前 `vllm` 未安装成功
  - 下一步需要处理代理/网络问题，或改用离线 wheel / 镜像源方案

### Session 25: 去代理后的 vLLM 修复推进

- 操作：
  - 检查环境变量，发现：
    - `HTTP_PROXY=http://127.0.0.1:17890`
    - `HTTPS_PROXY=http://127.0.0.1:17890`
  - 确认本地代理当前不可用，是导致 `pip` 失败的直接原因
  - 在不带代理的环境下成功升级：
    - `sentry-sdk -> 2.55.0`
  - 确认 `fastapi-cloud-cli` 与 `fastapi-cli` 已可用
  - 使用 `pip install --no-deps vllm==0.18.0` 补装 `vllm` 主包
- 结果：
  - `importlib.util.find_spec('vllm')` 成功
  - `import vllm` 成功，首次导入耗时约 `99.42s`
  - `api_server` 入口仍然很慢，尚未完成最终服务级验证
- 额外修复：
  - 已在 `deployment/serve_vllm.sh` 中显式 `unset HTTP_PROXY/HTTPS_PROXY`
- 结果：
  - DPO 在训练开始阶段失败
  - 报错位置：`training/run_dpo.py`
  - 主要错误：
    - `RuntimeError: Expected all tensors to be on the same device`
    - `ref_model` 的 embedding 权重仍在 CPU，而输入 tensor 在 `cuda:0/1`
- 处理：
  - 在 `run_dpo.py` 中引入 `LOCAL_RANK`
  - 将 `policy_model` 和 `ref_model` 显式移动到对应本地 GPU
  - 同时关闭 `use_cache`

### Session 17: DPO 数据质量诊断

- 操作：
  - 检查 `data/processed/dpo_candidates.json`
  - 检查 `data/processed/dpo_pairs.json`
- 结果：
  - 原始候选总数：`2048`
  - 非垃圾候选代理计数仅约：`205`
  - 原始 pair 数：`187`
  - 按更严格规则过滤后，可用 pair 代理计数仅约：`59`
- 典型问题：
  - 空串
  - 单独的 `assistant`
  - 单独的 `<tool_call>`
  - 单独的 `recovered`
- 结论：
  - “直接从当前 SFT 模型采样 DPO 候选”在当前阶段不可用
  - 若继续使用这批数据，会显著降低 DPO 结果可信度

### Session 18: DPO 数据生成降级决策

- 决策背景：
  - 项目要求优先从 SFT 模型采样候选
  - 但当前 SFT 候选质量过低，过滤后几乎无法形成可训练偏好对
- 降级方案：
  - 改为使用 baseline `Qwen3-8B` 生成 DPO 候选
  - 仍以 `runs/sft_qwen3_lora` 作为 DPO policy 初始化
  - 记录为“偏好数据生成源”的工程降级，而非放弃 DPO
- 已执行：
  - 停止低收益的 SFT-based DPO 数据重建会话
  - 在 `tmux` 中启动 baseline-based DPO 数据重建会话 `emocalcu_rebuild_dpo_baseline_gpu12`

### Session 19: baseline-based DPO 数据重建完成

- 操作：
  - 使用 `models/Qwen3-8B` 而非 `runs/sft_qwen3_lora` 生成 DPO 候选
  - 运行输出文件：
    - `data/processed/dpo_candidates_baseline.json`
    - `data/processed/dpo_pairs_baseline.json`
- 结果：
  - prompt 数：`512`
  - pair 数：`354`
  - 每个 prompt 平均候选数：`4.0`
  - 平均尝试次数约：`4.04`
- 抽样观察：
  - 当前 pair 质量明显优于 SFT-source 版本
  - 代理有效 pair 计数为 `354/354`
  - 仍存在少量风格差异不算特别大，但整体已达到可训练水平

### Session 20: baseline-pair 驱动的 DPO 重启

- 操作：
  - 更新 `scripts/run_dpo_gpu56.sh`，使其支持从命令行传入 `train-file` 与 `output-dir`
  - 在 `tmux` 会话 `emocalcu_dpo_gpu56` 中重新启动 DPO
  - 训练数据改为 `data/processed/dpo_pairs_baseline.json`
  - 输出目录改为 `runs/dpo_qwen3_baseline_pairs`
- 当前状态：
  - tmux 会话已启动
  - 启动日志已出现 `torchrun` 初始化输出
  - 仍需继续观察 loss、checkpoint 和最终产物是否正常落盘

### Session 26: 部署运行态复核

- 操作：
  - 重新核对 `PROJECT_STATUS.md`、`README.md`、`reports/final_report.md`
  - 使用本地端口探测检查 `8001` 与 `7860` 的运行状态
  - 对 `vLLM` 服务执行 `curl http://127.0.0.1:8001/v1/models`
  - 对 `Gradio` 页面执行 `curl http://127.0.0.1:7860/`
  - 复核 `logs/vllm_gpu12.log`、`logs/vllm_gpu5.log`、`logs/demo_hf_gpu12.log`
- 结果：
  - `vllm 0.18.0` 已可导入
  - `fastapi 0.135.2` 已可导入
  - `http://127.0.0.1:8001/v1/models` 返回正常 JSON，服务模型名为 `models/Qwen3-8B`
  - `http://127.0.0.1:7860/` 返回完整 Gradio HTML，页面中显示后端为 `vLLM / OpenAI API`
  - 早先的 `GPU 1,2` 双卡 `vLLM` 失败原因已明确为显存不足
  - `GPU 5` 单卡版本在 `--enforce-eager --max-model-len 8192` 条件下可运行
  - `HF + Gradio` 兜底脚本当前仍失败，报错为：
    - `TypeError: Chatbot.__init__() got an unexpected keyword argument 'type'`
- 结论：
  - `vLLM + Gradio` 主路线已完成实机服务级验证
  - 当前项目主线已从“训练与部署打通”进入“最终自检与交付清单整理”

### Session 27: SFT/DPO v2 代码改造与实验重启

- 操作：
  - 新增 `training/qwen3_no_think.py`，实现手工 `Qwen3 no-think` 序列化
  - 新增 `training/quality_rules.py`，统一质量打分、junk 判定、think leak 判定与命令式惩罚
  - 重写 `training/run_sft.py`，改为 assistant-only loss，并支持 v2 LoRA 配置与 `max_steps`
  - 修复 `training/run_dpo.py` 的 prompt 构造，切到 no-think serializer，并下调默认 `beta / lr`
  - 升级 `training/build_dpo_pairs.py` 为多候选文件输入 + margin 过滤
  - 新增 `training/prepare_sft_v2_data.py`
  - 新增扩展评测集脚本与场景库：
    - `evaluation/build_extended_eval_set.py`
    - `evaluation/typical_support_scenarios.json`
    - `evaluation/edge_case_scenarios.json`
  - 新增 v2 运行脚本：
    - `scripts/build_sft_v2_data_gpu12.sh`
    - `scripts/run_sft_v2_gpu56.sh`
    - `scripts/build_dpo_v2_pairs_gpu12.sh`
    - `scripts/run_dpo_v2_gpu56.sh`
    - `scripts/run_model_eval_extended.sh`
- 静态验证：
  - `python3 -m py_compile` 已通过
  - `evaluation/build_extended_eval_set.py` 已成功生成 `100` 条固定评测集
  - `training/prepare_sft_v2_data.py --rewrite-limit 0` 已成功生成 smoke 版数据：
    - `train = 31066`
    - `validation = 1500`
- 关键检查：
  - 抽样 tokenization 后，assistant 监督区不再包含 `<think>`
  - assistant 监督区也不再以 `assistant` 字面量开头
- 运行态：
  - 已在 `tmux` 中启动完整 v2 数据重建：`emocalcu_build_sft_v2_gpu2`
  - 已在 `tmux` 中启动 `50 step` SFT smoke：`emocalcu_sft_v2_smoke_gpu4`
  - `SFT smoke` 已产出 `runs/sft_qwen3_v2_smoke/checkpoint-25`
- 当前判断：
  - v2 主线已经从“方案设计”进入“真实运行验证”
  - 下一关键节点是 smoke 训练结束后的输出质量检查，以及完整 v2 数据重建的 rewrite 规模统计

### Session 28: CLI 交互入口与 v2 数据重建修正

- 现象：
  - 用户侧当前不便通过 `tmux` 直接查看实验终端输出
  - 首轮完整 `v2` 数据重建结束后，`summary.json` 显示：
    - `train = 886`
    - `validation = 1500`
    - `rewritten = 742`
  - 这说明训练集规模异常偏小，不符合原先预期
- 定位：
  - `prepare_sft_v2_data.py` 中验证集抽样直接取高分样本，导致高质量样本大量被抽走
  - 同时首轮 `gold_threshold / rewrite_threshold / rewrite_margin` 偏严
- 修复：
  - 将验证集抽样改为从高质量池中做确定性随机抽样，而不是直接截取 top-k
  - 将默认阈值调整为：
    - `gold_threshold = 1.0`
    - `rewrite_threshold = -0.25`
    - `rewrite_margin = 0.75`
  - 新增 CLI 交互脚本：
    - `demo/cli_chat.py`
    - `scripts/run_demo_cli_vllm.sh`
    - `scripts/run_demo_cli_hf.sh`
  - 将 CLI 入口接入现有高风险表达拦截逻辑
- 验证：
  - 重新运行 smoke 数据构建：
    - `train = 31066`
    - `validation = 1500`
    - 规模恢复正常
  - `printf '/quit\n' | bash scripts/run_demo_cli_vllm.sh` 已成功启动并退出，说明 CLI 入口可用
  - 已在 `tmux` 中重新启动修正版完整数据重建：
    - `emocalcu_build_sft_v2_gpu2_r2`
- 当前状态：
  - `CLI` 已成为正式可用的非 Web 交互路径
  - `v2` 数据主线已切换到修正版重建

### Session 29: v2_r2 运行态核对与负责人判断

- 操作：
  - 核对 `PROJECT_STATUS.md`、`comparison_summary.md`
  - 检查 `results/sft_qwen3_v2_smoke_baseline5/metrics.json`
  - 检查 `results/sft_qwen3_v2_smoke_baseline5/scenario_scores.json`
  - 检查 `data/processed/v2_r2/summary.json`
  - 检查 `logs/build_sft_v2_data_r2.log`
  - 检查 `tmux`、Python 进程和 GPU 占用
- 结果：
  - `v2_r2` 当前尚未完成，`data/processed/v2_r2/summary.json` 还不存在
  - 但 `prepare_sft_v2_data.py` 进程仍在运行，运行时长已超过 40 分钟
  - `GPU 2` 当前约占用 `18 GiB` 显存、利用率接近 `99%`，说明数据重建并未挂死
  - `SFT v2 smoke` 的 5 条快速评测已完成，关键指标为：
    - `junk_rate = 0.2`
    - `think_leak_rate = 0.0`
    - `too_short_rate = 0.2`
    - `avg_overall = 1.0`
  - 典型异常样例包括：
    - `.` 
    - `机器人`
    - `姐妹`
- 负责人判断：
  - 当前不能启动正式 `SFT v2`
  - 当前也不应继续 `DPO v2`
  - 现阶段第一优先仍然是等待 `v2_r2` 完成并执行数据审计
  - 即便 `no-think + assistant-only` 已经缓解旧污染，数据风格问题仍未解决，必须先在数据层处理

### Session 30: v2_r4_clean 轻量后处理与正式 SFT v2 启动

- 背景：
  - `v2_r4` 相比 `v2_r3` 已明显降低命令式、医疗化和结构化建议表达
  - 但审计中仍发现少量格式污染：
    - `<tool_response>`
    - `<tool_call>`
    - `咨询师：` 等角色前缀
  - 因此决定先做一次轻量后处理，再决定是否放行正式训练
- 实现：
  - 新增数据清洗脚本：
    - `training/clean_sft_dataset.py`
  - 新增清洗入口：
    - `scripts/build_v2_r4_clean.sh`
  - 清洗规则：
    - 去除 `<tool_response>`、`<tool_call>`
    - 去除回复开头的 `咨询师：/来访者：/支持者：/用户：/助理：`
    - 清洗后重新计算 `quality_score` 与 `quality_breakdown`
  - 正式启动脚本：
    - `scripts/run_sft_v2_r4_clean_gpu4.sh`
- 清洗结果：
  - `data/processed/v2_r4_clean/train_clean_summary.json`
    - `num_input_records = 8144`
    - `num_output_records = 8144`
    - `num_dropped_records = 0`
    - `tool_marker_records = 11`
    - `role_prefix_records = 56`
  - `data/processed/v2_r4_clean/validation_clean_summary.json`
    - `tool_marker_records = 2`
    - `role_prefix_records = 10`
- 审计结果：
  - `results/v2_r4_clean_audit_report.json`
  - 关键检查：
    - `<tool_response> = 0`
    - `<tool_call> = 0`
    - `咨询师：` 前缀污染 = 0
    - `imperative_hard = 0`
    - `medicalized = 0`
  - 仍存在少量软风格残留：
    - `persona_terms = 30`
    - 个别样本中仍可见 `宝宝/哥哥/姐妹/妹妹` 等称呼
  - 负责人判断：
    - 当前 `v2_r4_clean` 已消除会直接污染训练目标的硬问题
    - 虽仍存在轻微软风格残留，但已达到“可启动正式 SFT v2”的标准
- 训练启动：
  - 已在 `GPU 4` 上启动正式 `SFT v2`
  - `tmux` 会话：
    - `emocalcu_sft_v2_r4_clean_gpu4`
  - 日志：
    - `logs/sft_qwen3_v2_r4_clean_gpu4.log`
  - 输出目录：
    - `runs/sft_qwen3_v2_r4_clean_gpu4`
  - 早期训练状态：
    - `step 10 loss ≈ 0.9496`
    - `step 20 loss ≈ 0.4958`
    - `step 30 loss ≈ 0.3576`
    - `step 40 loss ≈ 0.3212`
    - `step 50 loss ≈ 0.3144`
    - 约在 `100/255 step` 时完成一次中间评估，`eval_loss ≈ 0.2957`
- 当前结论：
  - 第二轮优化已经从“数据修正阶段”进入“正式训练阶段”
  - 下一关键节点是等待 `SFT v2` 完成，然后统一重跑自动评测，判断是否真正接近或超过 baseline

### Session 31: 正式 SFT v2 完成

- 完成情况：
  - `v2_r4_clean` 上的正式 `SFT v2` 已完成
  - 输出目录：
    - `runs/sft_qwen3_v2_r4_clean_gpu4`
  - 关键产物：
    - `checkpoint-100`
    - `checkpoint-200`
    - `checkpoint-255`
    - `adapter_model.safetensors`
    - `eval_metrics.json`
- 最终验证指标：
  - `eval_loss = 0.272341251373291`
  - `eval_runtime = 90.5596`
  - `eval_samples_per_second = 16.564`
  - `eval_steps_per_second = 8.282`
  - `epoch = 1.0`
- 训练过程观察：
  - 中间评估时，`epoch ≈ 0.3929` 的 `eval_loss ≈ 0.2957`
  - 后续在 `epoch ≈ 0.7859` 进一步下降到 `eval_loss ≈ 0.2757`
  - 最终收敛到 `0.2723`，说明从训练稳定性看本轮优化是健康的
- 当前边界：
  - 以上仅说明训练与验证损失表现正常
  - 是否真正优于 baseline，仍必须以统一自动评测结果为准
- 下一步：
  - 立即对 `SFT v2` 重跑自动评测
  - 与 `baseline / 第一轮 SFT / 第一轮 DPO` 做统一对比

### Session 32: SFT v2 quick eval 结果与 DPO v2 放行

- 评测范围：
  - 使用 `25` 条快速评测集：
    - `20` 条 `heldout_smile`
    - `5` 条 `edge_case`
  - 对比对象：
    - `baseline Qwen3-8B`
    - `SFT v2 (runs/sft_qwen3_v2_r4_clean_gpu4)`
- 结果汇总：
  - `results/comparison_qwen3_v2_quick/comparison_summary.md`
  - baseline：
    - `avg_overall = 2.76`
    - `avg_empathy = 1.96`
    - `avg_supportiveness = 2.08`
    - `avg_safety = 4.56`
    - `junk_rate = 0.16`
    - `too_short_rate = 0.04`
  - `SFT v2`：
    - `avg_overall = 3.12`
    - `avg_empathy = 2.12`
    - `avg_supportiveness = 2.20`
    - `avg_safety = 4.84`
    - `junk_rate = 0.08`
    - `too_short_rate = 0.0`
- 观察：
  - `SFT v2` 首次在统一快速评测上超过 baseline
  - 提升最明显的是：
    - `avg_overall`
    - `avg_safety`
    - `junk_rate`
  - 代价是：
    - `distinct_1/2` 略低于 baseline
    - `avg_exploration` 略低（`1.0` vs `1.08`）
- 负责人判断：
  - 这轮结果已经达到“可继续进入 `DPO v2`”的门槛
  - 当前不再停留在“数据修正是否有效”的阶段，而是进入“在已有正向 SFT 基础上继续做偏好对齐”的阶段
  - 但仍不能宣称最终模型已全面反超 baseline，因为：
    - 当前仍缺更大规模扩展评测
    - 当前仍缺 `MT-Bench/MMLU` 子集补证
- 下一步：
  - 启动 `DPO v2` 的候选生成与 pair 构建
  - 后续对 `DPO v2` 做同口径自动评测

### Session 33: DPO v2 重启与自动接力训练

- 背景：
  - 首次 `DPO v2` 构建会话在权重加载后没有留下可观测的产物，且会话内已无活动进程
  - 为避免黑盒式等待，对候选生成脚本做了可观测性增强，并将构建规模调整为更稳妥的 `512` 条 seed prompt
- 工程调整：
  - 更新 `training/generate_dpo_candidates.py`
    - 新增 `--progress-every`
    - 支持写出 `.partial` 中间文件
    - 每处理一批 prompt 打印 `processed / total`
  - 新增重启脚本：
    - `scripts/build_dpo_v2_r4clean_gpu6.sh`
  - 新增自动接力训练脚本：
    - `scripts/wait_and_run_dpo_v2_r4clean_gpu67.sh`
- 当前运行态：
  - 构建会话：
    - `emocalcu_build_dpo_v2_r4clean_gpu6`
  - 日志：
    - `logs/build_dpo_v2_pairs_r4_clean.log`
  - 自动接力训练会话：
    - `emocalcu_wait_run_dpo_v2_gpu67`
  - 训练等待日志：
    - `logs/wait_run_dpo_v2_r4_clean_gpu67.log`
- 已确认进展：
  - `DPO v2` 候选生成已经开始打印进度
  - 当前至少已处理到：
    - `processed = 25 / 512`
    - 输出中间文件目标：
      - `data/processed/v2_r4_clean_dpo/dpo_candidates_baseline_v2.json.partial`
- 当前判断：
  - `DPO v2` 已经从“计划中”推进为“真实运行中”
  - 一旦 `dpo_pairs_v2.json` 生成，训练将自动在 `GPU 6,7` 上接力启动，不需要人工再盯一次

### Session 34: 偏好数据完成并切换到 GPU 3,4 训练

- 偏好数据完成情况：
  - `data/processed/v2_r4_clean_dpo/dpo_candidates_baseline_v2.json`
  - `data/processed/v2_r4_clean_dpo/dpo_candidates_sft_v2.json`
  - `data/processed/v2_r4_clean_dpo/dpo_pairs_v2.json`
  - 最终 `pair` 数量：
    - `303`
- 调度调整：
  - 虽然此前已在 `GPU 6,7` 上自动启动过一轮 `DPO v2`
  - 但根据最新要求，主训练改为切换到 `GPU 3,4`
- 当前训练态：
  - `tmux` 会话：
    - `emocalcu_dpo_v2_gpu34`
  - 日志：
    - `logs/dpo_qwen3_v2_gpu34.log`
  - 输出目录：
    - `runs/dpo_qwen3_v2_r4_clean_gpu34`
- 已确认进度：
  - `GPU 3,4` 均已加载模型并进入训练
  - 当前已推进到至少：
    - `step 5 / 19`
  - 早期日志：
    - `loss ≈ 0.6937`
    - `loss ≈ 0.6927`
    - `loss ≈ 0.6912`
- 当前判断：
  - 主实验链已经按最新要求切换到 `GPU 3,4`
  - 现在应继续等待 `DPO v2` 训练完成，再统一做最终模型对比

### Session 35: DPO v2 训练完成

- 训练完成情况：
  - `DPO v2` 已在 `GPU 3,4` 上完成
  - 输出目录：
    - `runs/dpo_qwen3_v2_r4_clean_gpu34`
  - 关键 checkpoint：
    - `checkpoint-19`
- 训练日志摘要：
  - 日志：
    - `logs/dpo_qwen3_v2_gpu34.log`
  - 训练总步数：
    - `19`
  - 最终：
    - `train_loss ≈ 0.6907`
    - `train_runtime ≈ 161.9s`
    - `rows = 303`
- 当前边界：
  - 训练完成并不等同于最终效果优于 `SFT v2`
  - 仍需通过统一生成式评测，才能决定是否把最终模型切换为 `DPO v2`
- 下一步：
  - 对 `DPO v2` 重跑 quick eval / 最终对比评测
  - 与 `baseline / 第一轮 SFT / 第一轮 DPO / SFT v2` 做统一对比

### Session 36: DPO v2 最终 quick eval 与当前最优模型更新

- 评测范围：
  - 对 `DPO v2` 使用与 `baseline / SFT v2` 完全相同的 `quick_eval_q25`
  - 结果目录：
    - `results/quick_eval_dpo_v2_q25`
  - 三方汇总：
    - `results/comparison_qwen3_v2_final/comparison_summary.md`
- 核心结果：
  - baseline：
    - `avg_overall = 2.76`
    - `avg_supportiveness = 2.08`
    - `avg_safety = 4.56`
    - `junk_rate = 0.16`
  - `SFT v2`：
    - `avg_overall = 3.12`
    - `avg_supportiveness = 2.20`
    - `avg_safety = 4.84`
    - `junk_rate = 0.08`
  - `DPO v2`：
    - `avg_overall = 3.44`
    - `avg_supportiveness = 2.48`
    - `avg_safety = 4.92`
    - `junk_rate = 0.04`
- 观察：
  - `DPO v2` 在当前同口径 quick eval 上优于 baseline 与 `SFT v2`
  - 提升主要体现在：
    - `avg_overall`
    - `avg_supportiveness`
    - `avg_safety`
    - `junk_rate`
  - 代价是：
    - `distinct_1/2` 继续略低于 baseline
    - `avg_exploration` 与 `SFT v2` 持平，仍未高于 baseline
- 当前结论：
  - 在当前阶段、当前评测口径下，默认最优模型已从 `SFT v2` 更新为 `DPO v2`
  - 这意味着主线目标“将模型从低于 baseline 拉回并进一步优于 baseline”已经在当前自动评测上达成
- 结论边界：
  - 当前结论仍基于 `quick_eval_q25` 与代理评分
  - 尚未补跑 `MT-Bench/MMLU`，因此不能宣称“通用能力已被完整验证为未显著下降”

### Session 37: 最终收尾与交付文档更新

- 本轮完成的文档收尾：
  - 更新 `README.md`
  - 重写 `reports/final_report.md`
  - 新增 `docs/final_delivery.md`
  - 同步 `PROJECT_STATUS.md`
- 当前统一结论：
  - 第一轮正式 `SFT/DPO` 未超过 baseline
  - 第二轮 `SFT v2` 已超过 baseline
  - 第二轮 `DPO v2` 在 `quick_eval_q25` 上进一步优于 `SFT v2`
- 当前默认最终模型：
  - `runs/dpo_qwen3_v2_r4_clean_gpu34`
- 当前推荐 demo 路线：
  - `HF backend + DPO v2 adapter`
- 仍保留的边界：
  - `MT-Bench/MMLU` 尚未补跑
  - 当前结论仍主要基于项目内自动代理评测
