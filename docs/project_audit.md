# 项目现状审计

## 审计时间

- 日期：2026-03-19

## 仓库现状

- 当前目录在启动时为空目录
- 无现有代码、无 README、无环境文件、无配置、无数据目录
- 无 git 初始化，已由本项目首次初始化

## 运行环境

- 操作系统：Ubuntu Linux 5.4.0-125-generic
- CPU：64 vCPU，Intel Xeon Platinum 8358P
- 系统内存：约 2.0 TiB
- 可用工作目录：`/9950backfile/heruxuan/emocalcu`
- 磁盘：
  - `/9950backfile` 可用空间约 697 TiB
  - `/tmp` 可用空间约 445 GiB
  - `/public` 几乎满载，不适合作为新增产物目录

## Python 与 conda

- 默认 `python` 指向 Python 2.7.18
- 默认 `python3` 指向 Python 3.10.4
- shell 未自动加载 conda
- 已定位 conda：`/9950backfile/heruxuan/path/miniconda3`
- 已发现环境：
  - `base`
  - `path`

## `path` 环境审计

- Python：3.10.19
- 已安装：
  - `torch 2.10.0+cu128`
  - `transformers 5.3.0`
  - `datasets 4.6.1`
  - `deepspeed 0.18.7`
  - `accelerate 1.13.0`
  - `flash_attn 2.8.3`
- 缺失：
  - `peft`
  - `trl`
  - `gradio`
  - `vllm`
  - `llamafactory`

## GPU 现状

- 在当前沙箱中：
  - `nvidia-smi` 失败
  - `torch.cuda.is_available()` 返回 `False`
  - `torch.cuda.device_count()` 返回 `0`
- 在脱离沙箱的 conda 环境中：
  - `torch.cuda.is_available()` 返回 `True`
  - `torch.cuda.device_count()` 返回 `8`
  - 设备名称为 `NVIDIA A800-SXM4-80GB`
- 结论：
  - 真实硬件可用
  - 课程项目后续默认绑定 `GPU 5,6`

## 本地缓存

- 已发现 Hugging Face cache 根目录：`/public/heruxuan/.cache/huggingface/hub`
- 当前可见缓存非常有限：
  - 模型缓存中检测到 `mistralai--Mistral-7B-v0.1`
  - 数据缓存中检测到 `HuggingFaceFW--fineweb-edu`
- 暂未发现 `Qwen3-8B-Instruct` 的本地缓存

## 数据源审计

- 已验证 `thu-coai/esconv` 可下载，但 Hugging Face 版本内容为英文情绪支持对话
- 已下载 GitHub 公开仓库 `qiuhuachuan/smile`
- 当前采用 `SmileChat` 作为主数据源，原因：
  - 中文为主
  - 明确面向心理健康支持对话
  - 公共可获得
  - 可直接转换为多轮 SFT 样本

## 当前数据处理结果

- 数据源：SmileChat
- 原始文件数：6796
- 生成 SFT 样本数：37227
- 过滤 turn 数：632
- 切分：
  - train: 29781
  - validation: 3723
  - test: 3723
- 基础测试集代理指标：
  - distinct-1: 0.0063
  - distinct-2: 0.1346
  - 平均回复长度: 100.61
  - 负向行为频率: 0.0086

## 基座模型审计

- 用户目标中写为 `Qwen3-8B-Instruct`
- 实际在 Hugging Face 可查询到的公开仓库名为 `Qwen/Qwen3-8B`
- `Qwen/Qwen3-8B` 的标签包含 `conversational`、`text-generation`，可作为当前 baseline

## 建议执行路线

1. 补齐 `peft`、`trl`、`gradio`，并尝试安装 `LLaMA-Factory`。
2. 若 `LLaMA-Factory` 安装或兼容性受阻，则切换到 `Transformers + PEFT + TRL`。
3. 下载公开中文共情/心理支持数据，清洗并统一为 Alpaca 风格。
4. 跑通 baseline 推理样例后，再进入 SFT。
5. 基于本地启发式或 judge 构造偏好数据，执行小规模 DPO 起步。
6. 完成自动评测、可视化、demo 与报告。

## 当前风险

1. GPU 真实可见性未确认。
2. 关键依赖尚未补齐。
3. Qwen3 模型与目标数据集需要额外下载。
4. 用户指定“使用 GPU 5 6”，但当前进程可见 GPU 拓扑尚未完成核验，需待实际 `CUDA_VISIBLE_DEVICES` 配置后落实。
