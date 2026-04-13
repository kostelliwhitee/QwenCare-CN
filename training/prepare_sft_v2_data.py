#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.qwen3_no_think import DEFAULT_SYSTEM_PROMPT, build_generation_inputs, clean_response
from training.quality_rules import (
    has_style_drift,
    has_think_leak,
    is_junk_response,
    overall_score,
    passes_keep_gate,
    quality_breakdown,
)


HIGH_RISK_PATTERNS = [
    "自杀",
    "不想活",
    "结束生命",
    "杀人",
    "同归于尽",
]


def load_json_list(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def contains_high_risk(text: str) -> bool:
    return any(pattern in (text or "") for pattern in HIGH_RISK_PATTERNS)


def build_bad_words_ids(tokenizer) -> list[list[int]]:
    groups = []
    for pattern in ["<think>", "</think>"]:
        token_ids = tokenizer.encode(pattern, add_special_tokens=False)
        if token_ids:
            groups.append(token_ids)
    return groups


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


def rewrite_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs, prompt_length = build_generation_inputs(tokenizer, DEFAULT_SYSTEM_PROMPT, prompt, model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.8,
            do_sample=True,
            bad_words_ids=build_bad_words_ids(tokenizer) or None,
        )
    raw = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return clean_response(raw).strip()


def stable_key(record: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "instruction": record.get("instruction", ""),
                "input": record.get("input", ""),
                "output": record.get("output", ""),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def normalize_record(record: dict) -> dict:
    return {
        "id": record.get("id") or stable_key(record)[:12],
        "dataset": record.get("dataset", "unknown"),
        "system": record.get("system") or DEFAULT_SYSTEM_PROMPT,
        "instruction": record.get("instruction", "").strip(),
        "input": record.get("input", "").strip(),
        "output": clean_response(record.get("output", "")).strip(),
    }


def should_drop(record: dict) -> bool:
    output = record["output"]
    if not output:
        return True
    if contains_high_risk(record["instruction"]) or contains_high_risk(output):
        return True
    if has_think_leak(output):
        return True
    if has_style_drift(output):
        return True
    if is_junk_response(output):
        return True
    return False


def select_validation(records: list[dict], validation_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    ranked = sorted(records, key=lambda item: (item["meta"]["quality_score"], item["id"]), reverse=True)
    if len(ranked) <= validation_size:
        return ranked, []
    pool_size = min(len(ranked), max(validation_size * 4, validation_size))
    pool = ranked[:pool_size]
    rng = random.Random(seed)
    val_ids = {item["id"] for item in rng.sample(pool, validation_size)}
    val = [item for item in ranked if item["id"] in val_ids]
    val_ids = {item["id"] for item in val}
    train = [item for item in ranked if item["id"] not in val_ids]
    return train, val


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rewrite-model-path", default="")
    parser.add_argument("--gold-threshold", type=float, default=1.0)
    parser.add_argument("--rewrite-threshold", type=float, default=-0.25)
    parser.add_argument("--rewrite-margin", type=float, default=0.75)
    parser.add_argument("--rewrite-limit", type=int, default=6000)
    parser.add_argument("--validation-size", type=int, default=1500)
    parser.add_argument("--dpo-seed-size", type=int, default=4096)
    parser.add_argument("--fallback-keep-threshold", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    seen = set()
    for input_file in args.input_file:
        for item in load_json_list(Path(input_file)):
            normalized = normalize_record(item)
            key = stable_key(normalized)
            if key in seen or should_drop(normalized):
                continue
            normalized["meta"] = {
                "quality_score": overall_score(normalized["output"]),
                "quality_breakdown": quality_breakdown(normalized["output"]),
                "source_type": "gold",
                "rewritten_from": "",
                "original_quality_score": overall_score(normalized["output"]),
            }
            records.append(normalized)
            seen.add(key)

    model = None
    tokenizer = None
    rewrite_count = 0
    if args.rewrite_model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.rewrite_model_path, trust_remote_code=True)
        model = load_model(args.rewrite_model_path)

    upgraded = []
    dropped_for_low_quality = 0
    for idx, record in enumerate(records, start=1):
        quality_score = record["meta"]["quality_score"]
        can_keep_original = quality_score >= args.fallback_keep_threshold and passes_keep_gate(record["output"])

        if quality_score >= args.gold_threshold and passes_keep_gate(record["output"]):
            upgraded.append(record)
        elif model is None or rewrite_count >= args.rewrite_limit:
            if can_keep_original:
                upgraded.append(record)
            else:
                dropped_for_low_quality += 1
        elif quality_score < args.rewrite_threshold:
            dropped_for_low_quality += 1
        else:
            rewritten = rewrite_response(model, tokenizer, record["instruction"])
            rewritten_score = overall_score(rewritten)
            rewritten_breakdown = quality_breakdown(rewritten)
            if is_junk_response(rewritten) or has_style_drift(rewritten) or not passes_keep_gate(rewritten):
                if can_keep_original:
                    upgraded.append(record)
                else:
                    dropped_for_low_quality += 1
            elif rewritten_score < quality_score + args.rewrite_margin:
                if can_keep_original:
                    upgraded.append(record)
                else:
                    dropped_for_low_quality += 1
            else:
                record = dict(record)
                record["output"] = rewritten
                record["meta"] = {
                    **record["meta"],
                    "quality_score": rewritten_score,
                    "quality_breakdown": rewritten_breakdown,
                    "source_type": "baseline_rewrite",
                    "rewritten_from": "baseline_qwen3",
                }
                upgraded.append(record)
                rewrite_count += 1

        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(records)):
            print(
                json.dumps(
                    {
                        "stage": "prepare_sft_v2_data",
                        "processed": idx,
                        "total": len(records),
                        "kept": len(upgraded),
                        "rewritten": rewrite_count,
                        "dropped_for_low_quality": dropped_for_low_quality,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    train_records, validation_records = select_validation(upgraded, args.validation_size, args.seed)
    dpo_seed = [
        {
            "id": item["id"],
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"],
            "system": item["system"],
            "meta": item["meta"],
        }
        for item in train_records[: args.dpo_seed_size]
    ]

    output_dir = Path(args.output_dir)
    save_json(output_dir / "train.json", train_records)
    save_json(output_dir / "validation.json", validation_records)
    save_json(output_dir / "dpo_seed_prompts.json", dpo_seed)
    save_json(
        output_dir / "summary.json",
        {
            "num_input_records": len(records),
            "num_train_records": len(train_records),
            "num_validation_records": len(validation_records),
            "num_rewritten_records": rewrite_count,
            "num_gold_records": sum(item["meta"]["source_type"] == "gold" for item in upgraded),
            "num_rewrite_records": sum(item["meta"]["source_type"] == "baseline_rewrite" for item in upgraded),
            "gold_threshold": args.gold_threshold,
            "rewrite_threshold": args.rewrite_threshold,
            "rewrite_margin": args.rewrite_margin,
            "fallback_keep_threshold": args.fallback_keep_threshold,
            "dropped_for_low_quality": dropped_for_low_quality,
            "rewrite_model_path": args.rewrite_model_path,
        },
    )
    print(
        json.dumps(
            {
                "train": len(train_records),
                "validation": len(validation_records),
                "rewritten": rewrite_count,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
