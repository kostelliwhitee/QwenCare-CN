#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "你是一位中文情绪支持助手。请提供温和、共情、非说教的回应。"
    "如果用户表达自伤、伤人或其他高风险意图，请优先建议联系现实中的紧急支持资源。"
    "不要输出思考过程、分析过程或<think>标签，只输出最终回复。"
)


def clean_response(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*思考[:：].*$", "", text, flags=re.MULTILINE)
    lines = text.splitlines()
    while lines:
        candidate = lines[0].strip()
        if not candidate:
            lines.pop(0)
            continue
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", candidate):
            lines.pop(0)
            continue
        break
    text = "\n".join(lines).strip()
    return text


def build_bad_words_ids(tokenizer) -> list[list[int]]:
    patterns = ["<think>", "</think>"]
    bad_words_ids = []
    for pattern in patterns:
        token_ids = tokenizer.encode(pattern, add_special_tokens=False)
        if token_ids:
            bad_words_ids.append(token_ids)
    return bad_words_ids


def load_model(model_name_or_path: str):
    model_path = Path(model_name_or_path)
    if (model_path / "adapter_config.json").exists():
        return AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )


def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output-file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = load_model(args.model_name_or_path)
    messages = build_messages(args.system_prompt, args.prompt)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = inputs.to(model.device)
    input_length = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            bad_words_ids=build_bad_words_ids(tokenizer) or None,
        )
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    response = clean_response(response) or response
    payload = {
        "model": args.model_name_or_path,
        "prompt": args.prompt,
        "response": response,
        "system_prompt": args.system_prompt,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output_file:
        path = Path(args.output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
