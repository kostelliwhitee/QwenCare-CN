#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from training.quality_rules import has_style_drift, has_think_leak, imperative_rate, is_junk_response


KEYWORDS = ["机器人", "姐妹", "妹妹", "亲亲", "宝宝", "宝贝", "哥哥", "assistant", "recovered"]


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_json(Path(args.input_file))
    counters = Counter()
    flagged_examples = []
    for idx, record in enumerate(records):
        output = (record.get("output", "") or "").strip()
        meta = record.get("meta", {})
        counters[f"source_type::{meta.get('source_type', 'unknown')}"] += 1
        if has_think_leak(output):
            counters["think_leak"] += 1
        if has_style_drift(output):
            counters["style_drift"] += 1
        if is_junk_response(output):
            counters["junk"] += 1
        if imperative_rate(output) > 0:
            counters["imperative"] += 1
        for keyword in KEYWORDS:
            if keyword in output:
                counters[f"keyword::{keyword}"] += 1
        if (
            has_style_drift(output)
            or is_junk_response(output)
            or imperative_rate(output) > 0
            or any(keyword in output for keyword in KEYWORDS)
        ) and len(flagged_examples) < 100:
            flagged_examples.append(
                {
                    "idx": idx,
                    "source_type": meta.get("source_type", "unknown"),
                    "output": output[:300],
                }
            )

    rng = random.Random(args.seed)
    sample = []
    sample_size = min(args.sample_size, len(records))
    for idx in sorted(rng.sample(range(len(records)), sample_size)):
        record = records[idx]
        sample.append(
            {
                "idx": idx,
                "source_type": (record.get("meta") or {}).get("source_type", "unknown"),
                "instruction": (record.get("instruction", "") or "")[:160],
                "output": (record.get("output", "") or "")[:320],
            }
        )

    payload = {
        "input_file": args.input_file,
        "num_records": len(records),
        "counters": dict(counters),
        "sample": sample,
        "flagged_examples": flagged_examples,
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_file": str(output_path), "num_records": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
