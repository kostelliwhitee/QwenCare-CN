#!/usr/bin/env python3
import re
from dataclasses import dataclass


DEFAULT_SYSTEM_PROMPT = (
    "你是一位中文情绪支持助手。请用温和、支持性、非说教式的方式回应用户。"
    "优先表达理解、接纳、陪伴、澄清和支持。避免冷漠、命令式指责和武断诊断。"
)

NO_THINK_TAG = "/no_think"


def ensure_no_think(text: str) -> str:
    text = (text or "").strip()
    if NO_THINK_TAG in text:
        return text
    if text:
        return f"{text}\n{NO_THINK_TAG}"
    return NO_THINK_TAG


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_response(text: str) -> str:
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*思考[:：].*$", "", text, flags=re.MULTILINE)
    lines = text.splitlines()
    while lines:
        head = lines[0].strip()
        if not head:
            lines.pop(0)
            continue
        if head.lower() in {"assistant", "recovered", "oplayer", "ypse"}:
            lines.pop(0)
            continue
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", head):
            lines.pop(0)
            continue
        break
    return "\n".join(lines).strip()


def render_chat(messages: list[dict], add_generation_prompt: bool = False, no_think: bool = True) -> str:
    chunks: list[str] = []
    for index, message in enumerate(messages):
        role = message["role"].strip()
        content = normalize_text(message.get("content", ""))
        if role == "system" and no_think:
            content = ensure_no_think(content)
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt and index == len(messages) - 1:
            chunks.append("<|im_start|>assistant\n")
    return "".join(chunks)


def build_chat_messages(system: str, user: str, assistant: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return messages


def response_suffix() -> str:
    return "<|im_end|>\n"


@dataclass
class TokenizedConversation:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    assistant_mask: list[int]


def tokenize_conversation(tokenizer, system: str, user: str, assistant: str, max_seq_length: int) -> TokenizedConversation:
    prompt_text = render_chat(
        build_chat_messages(system=system, user=user),
        add_generation_prompt=True,
        no_think=True,
    )
    completion_text = normalize_text(assistant) + response_suffix()
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + completion_ids)[:max_seq_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    attention_mask = [1] * len(input_ids)
    labels = input_ids[:]
    assistant_mask = [0] * len(input_ids)
    for index in range(prompt_length):
        labels[index] = -100
    for index in range(prompt_length, len(input_ids)):
        assistant_mask[index] = 1
    return TokenizedConversation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        assistant_mask=assistant_mask,
    )


def build_generation_inputs(tokenizer, system: str, user: str, device):
    prompt_text = render_chat(
        build_chat_messages(system=system, user=user),
        add_generation_prompt=True,
        no_think=True,
    )
    encoded = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    return encoded.to(device), encoded["input_ids"].shape[-1]
