#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.qwen3_no_think import DEFAULT_SYSTEM_PROMPT, build_generation_inputs, clean_response
from training.quality_rules import is_junk_response


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


def sample_once(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs, input_length = build_generation_inputs(tokenizer, DEFAULT_SYSTEM_PROMPT, prompt, model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            bad_words_ids=build_bad_words_ids(tokenizer) or None,
        )
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    cleaned = clean_response(response)
    return cleaned or response


def write_payload(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-attempts-per-prompt", type=int, default=12)
    parser.add_argument("--temperatures", default="0.2,0.6")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    prompts = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = load_model(args.model_name_or_path)

    output = []
    output_path = Path(args.output_file)
    partial_output_path = output_path.with_suffix(output_path.suffix + ".partial")
    total = len(prompts)
    for idx, item in enumerate(prompts, start=1):
        candidates = []
        temperatures = [float(token.strip()) for token in args.temperatures.split(",") if token.strip()]
        prompt = item["instruction"] if "instruction" in item else item["prompt"]
        if item.get("output") and not is_junk_response(item["output"]):
            candidates.append(item["output"].strip())
        attempts = 0
        while len(candidates) < args.num_candidates and attempts < args.max_attempts_per_prompt:
            temperature = temperatures[attempts % len(temperatures)]
            top_p = 0.8 if attempts % 2 == 0 else 0.9
            response = sample_once(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            attempts += 1
            if is_junk_response(response):
                continue
            if response in candidates:
                continue
            candidates.append(response)
        row = (
            {
                "id": item.get("id", f"row-{len(output)}"),
                "prompt": prompt,
                "candidates": candidates,
                "meta": {"attempts": attempts},
            }
        )
        output.append(row)
        if idx % args.progress_every == 0 or idx == total:
            write_payload(output, partial_output_path)
            print(
                json.dumps(
                    {
                        "stage": "generate_dpo_candidates",
                        "processed": idx,
                        "total": total,
                        "output_file": str(partial_output_path),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    write_payload(output, output_path)
    print(json.dumps({"rows": len(output), "output_file": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
