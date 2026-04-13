#!/usr/bin/env python3
import os

import gradio as gr

from demo.backend import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SYSTEM_PROMPT,
    backend_label,
    chat_once,
)


def chat_fn(message: str, history: list[dict], system_prompt: str, temperature: float, max_tokens: int):
    payload = chat_once(message, history, system_prompt, temperature, max_tokens)
    return payload["history"], payload["history"]


def render_chat_history(history: list[dict]) -> list[list[str]]:
    rendered = []
    for item in history:
        rendered.append([item.get("user", ""), item.get("assistant", "")])
    return rendered


def submit(message, history, prompt, temp, tokens):
    pairs, state_payload = chat_fn(message, history, prompt, temp, tokens)
    return render_chat_history(pairs), state_payload, ""


model_label = DEFAULT_ADAPTER_PATH or DEFAULT_MODEL_PATH

with gr.Blocks(title="中文 AI 心理医生 Demo") as demo:
    gr.Markdown(
        (
            "# 中文 AI 心理医生 Demo\n"
            "本系统仅用于课程项目与情绪支持研究演示，不构成专业医疗诊断或治疗建议。\n"
            "如涉及自伤、伤人、极端风险或紧急危机，请优先联系现实中的紧急支持资源。"
        )
    )
    gr.Markdown(f"当前后端：`{backend_label()}`  |  当前模型：`{model_label}`")

    with gr.Row():
        system_prompt = gr.Textbox(label="System Prompt", value=DEFAULT_SYSTEM_PROMPT, lines=4)
        with gr.Column():
            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.2, value=0.7, step=0.1)
            max_tokens = gr.Slider(label="Max Tokens", minimum=64, maximum=512, value=256, step=32)

    chatbot = gr.Chatbot(label="对话")
    state = gr.State([])
    user_input = gr.Textbox(label="输入", placeholder="例如：最近总觉得很焦虑，不知道该怎么办。")

    gr.Examples(
        examples=[
            ["我这几天总睡不好，一想到考试就心跳很快。"],
            ["分手以后我一直觉得是不是自己不值得被喜欢。"],
            ["最近工作压力特别大，我每天都在怀疑自己。"],
            ["我现在特别想伤害自己，感觉撑不住了。"],
        ],
        inputs=user_input,
    )

    user_input.submit(
        submit,
        [user_input, state, system_prompt, temperature, max_tokens],
        [chatbot, state, user_input],
    )

demo.queue(default_concurrency_limit=2)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
