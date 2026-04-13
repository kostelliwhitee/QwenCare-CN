#!/usr/bin/env python3
import os
import re
import threading
from pathlib import Path

import requests
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.qwen3_no_think import build_generation_inputs, clean_response

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = os.environ.get("DEMO_MODEL_PATH", str(PROJECT_ROOT / "models" / "Qwen3-8B"))
DEFAULT_ADAPTER_PATH = os.environ.get("DEMO_ADAPTER_PATH", "")
DEFAULT_API_BASE = os.environ.get("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
DEFAULT_BACKEND = os.environ.get("DEMO_BACKEND", "hf")
DEFAULT_SYSTEM_PROMPT = (
    "你是一位中文情绪支持助手。请给出温和、支持性、非说教式的回应，"
    "尽量先理解和接住用户情绪，再给出简短建议。"
    "不要输出思考过程、分析过程或<think>标签。"
)
SAFETY_RESPONSE = (
    "听起来你现在正处在很难受、也可能很危险的状态。"
    "我不能替代现实中的紧急帮助，但很希望你马上联系身边可信任的人，"
    "或尽快联系当地心理危机热线、急救电话、医院急诊。"
    "如果你愿意，也可以先告诉我你现在是否一个人，以及身边有没有可以立刻联系的人。"
)
HIGH_RISK_PATTERNS = [
    r"自杀",
    r"不想活",
    r"结束生命",
    r"伤害自己",
    r"割腕",
    r"跳楼",
    r"轻生",
    r"杀人",
    r"伤人",
    r"同归于尽",
]

_BACKEND = None
_BACKEND_LOCK = threading.Lock()


def build_bad_words_ids(tokenizer) -> list[list[int]]:
    patterns = ["<think>", "</think>"]
    token_groups = []
    for pattern in patterns:
        token_ids = tokenizer.encode(pattern, add_special_tokens=False)
        if token_ids:
            token_groups.append(token_ids)
    return token_groups


def contains_high_risk(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in HIGH_RISK_PATTERNS)


class HFBackend:
    def __init__(self, model_path: str, adapter_path: str = "") -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.tokenizer = None
        self.model = None
        self.lock = threading.Lock()

    def _load(self) -> None:
        if self.model is not None:
            return
        target_path = self.adapter_path if self.adapter_path else self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if self.adapter_path:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.adapter_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        self.model.eval()

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        with self.lock:
            self._load()
            system_prompt = messages[0]["content"]
            dialogue_lines = []
            for item in messages[1:]:
                role = item["role"]
                content = item["content"]
                if role == "user":
                    dialogue_lines.append(f"用户：{content}")
                elif role == "assistant":
                    dialogue_lines.append(f"助手：{content}")
            user_text = "\n".join(dialogue_lines[:-1] + [dialogue_lines[-1].replace("用户：", "", 1)]) if dialogue_lines else ""
            inputs, input_length = build_generation_inputs(
                self.tokenizer,
                system=system_prompt,
                user=user_text,
                device=self.model.device,
            )
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=temperature > 0,
                    bad_words_ids=build_bad_words_ids(self.tokenizer) or None,
                )
            text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            return clean_response(text) or text


class VLLMBackend:
    def __init__(self, api_base: str, model_name: str) -> None:
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        response = requests.post(
            f"{self.api_base}/chat/completions",
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=300,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        return clean_response(text) or text


def get_backend():
    global _BACKEND
    if _BACKEND is None:
        with _BACKEND_LOCK:
            if _BACKEND is None:
                if DEFAULT_BACKEND == "vllm":
                    _BACKEND = VLLMBackend(DEFAULT_API_BASE, DEFAULT_MODEL_PATH)
                else:
                    _BACKEND = HFBackend(DEFAULT_MODEL_PATH, DEFAULT_ADAPTER_PATH)
    return _BACKEND


def build_messages(message: str, history: list[dict], system_prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        user_text = turn.get("user")
        assistant_text = turn.get("assistant")
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": message})
    return messages


def chat_once(message: str, history: list[dict], system_prompt: str, temperature: float, max_tokens: int):
    message = (message or "").strip()
    history = history or []
    if not message:
        return {"response": "", "history": history}
    if contains_high_risk(message):
        history = history + [{"user": message, "assistant": SAFETY_RESPONSE}]
        return {"response": SAFETY_RESPONSE, "history": history, "risk_flagged": True}

    backend = get_backend()
    messages = build_messages(message, history, system_prompt)
    response = backend.chat(messages, temperature, max_tokens)
    history = history + [{"user": message, "assistant": response}]
    return {"response": response, "history": history, "risk_flagged": False}


def backend_label() -> str:
    return "vLLM / OpenAI API" if DEFAULT_BACKEND == "vllm" else "Transformers 本地推理"
