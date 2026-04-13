# tmux Runbook

## 约定

- 所有长任务都在 `tmux` 中启动
- 项目根目录：`/9950backfile/heruxuan/emocalcu`
- 正式 `SFT v2`：`4`
- 正式 `DPO v2`：`3,4`
- 基座模型 demo / baseline：`1,2`

## 建议会话名

- `emocalcu_qwen3_download`
- `emocalcu_baseline`
- `emocalcu_sft_v2`
- `emocalcu_dpo_v2`
- `emocalcu_demo_hf`
- `emocalcu_demo_vllm`

## 常用命令

启动 Qwen3 下载：

```bash
tmux new-session -d -s emocalcu_qwen3_download 'cd /9950backfile/heruxuan/emocalcu && scripts/download_qwen3.sh'
```

启动 baseline：

```bash
tmux new-session -d -s emocalcu_baseline 'cd /9950backfile/heruxuan/emocalcu && CUDA_VISIBLE_DEVICES=1,2 scripts/run_baseline_eval.sh models/Qwen3-8B results/baseline_qwen3_gpu12'
```

启动正式 `SFT v2`：

```bash
tmux new-session -d -s emocalcu_sft_v2 'cd /9950backfile/heruxuan/emocalcu && CUDA_VISIBLE_DEVICES=4 bash scripts/run_sft_v2_r4_clean_gpu4.sh'
```

启动正式 `DPO v2`：

```bash
tmux new-session -d -s emocalcu_dpo_v2 'cd /9950backfile/heruxuan/emocalcu && CUDA_VISIBLE_DEVICES=3,4 bash scripts/run_dpo_v2_gpu56.sh data/processed/v2_r4_clean_dpo/dpo_pairs_v2.json runs/sft_qwen3_v2_r4_clean_gpu4 runs/dpo_qwen3_v2_r4_clean_gpu34'
```

启动最终模型 CLI demo：

```bash
tmux new-session -d -s emocalcu_demo_hf 'cd /9950backfile/heruxuan/emocalcu && CUDA_VISIBLE_DEVICES=1 DEMO_MODEL_PATH=/9950backfile/heruxuan/emocalcu/models/Qwen3-8B DEMO_ADAPTER_PATH=/9950backfile/heruxuan/emocalcu/runs/dpo_qwen3_v2_r4_clean_gpu34 bash scripts/run_demo_cli_hf.sh'
```

启动基座模型 `vLLM + Gradio` demo：

```bash
tmux new-session -d -s emocalcu_demo_vllm 'cd /9950backfile/heruxuan/emocalcu && VLLM_PORT=8001 CUDA_VISIBLE_DEVICES=1,2 bash scripts/run_demo_vllm_gpu12.sh'
```

查看会话：

```bash
tmux ls
```

附着会话：

```bash
tmux attach -t emocalcu_sft_v2
```

查看日志：

```bash
tail -f logs/sft_qwen3_v2_r4_clean_gpu4.log
tail -f logs/dpo_qwen3_v2_gpu34.log
```
