#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import re
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.qwen3_no_think import DEFAULT_SYSTEM_PROMPT, build_generation_inputs, clean_response


FALLBACK_SYSTEM_PROMPT = (
    "你是一位中文情绪支持助手。请直接输出给用户可见的最终回复。"
    "禁止输出思考过程、分析过程、<think>标签、草稿或解释。"
    "回复需要温和、支持性、非说教。"
)


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


def needs_regeneration(raw: str, cleaned: str) -> bool:
    if not cleaned:
        return True
    if raw.lstrip().startswith("<think>"):
        return True
    if len(cleaned) < 20:
        return True
    return False


def generate_once(model, tokenizer, system_prompt: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float, bad_words_ids: list[list[int]]) -> str:
    inputs, input_length = build_generation_inputs(tokenizer, system_prompt, prompt, model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            bad_words_ids=bad_words_ids or None,
        )
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()


def generate_response(model, tokenizer, system_prompt: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    bad_words_ids = build_bad_words_ids(tokenizer)
    response = generate_once(model, tokenizer, system_prompt, prompt, max_new_tokens, temperature, top_p, bad_words_ids)
    cleaned = clean_response(response)
    if not needs_regeneration(response, cleaned):
        return cleaned

    fallback_response = generate_once(
        model,
        tokenizer,
        FALLBACK_SYSTEM_PROMPT,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.8,
        bad_words_ids=bad_words_ids,
    )
    fallback_cleaned = clean_response(fallback_response)
    return fallback_cleaned or cleaned or response.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    prompts = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = load_model(args.model_name_or_path)

    outputs = []
    output_path = Path(args.output_file)
    partial_output_path = output_path.with_suffix(output_path.suffix + ".partial")
    total = len(prompts)
    for idx, item in enumerate(prompts, start=1):
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            system_prompt=args.system_prompt,
            prompt=item["prompt"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        outputs.append({"id": item["id"], "prompt": item["prompt"], "response": response})
        if idx % args.progress_every == 0 or idx == total:
            partial_output_path.parent.mkdir(parents=True, exist_ok=True)
            partial_output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "stage": "run_batch_inference",
                        "processed": idx,
                        "total": total,
                        "output_file": str(partial_output_path),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(outputs), "output_file": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
