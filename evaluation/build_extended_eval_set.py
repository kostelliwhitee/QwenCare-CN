#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from training.quality_rules import overall_score


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-file", default="data/processed/test.json")
    parser.add_argument("--typical-file", default="evaluation/typical_support_scenarios.json")
    parser.add_argument("--edge-file", default="evaluation/edge_case_scenarios.json")
    parser.add_argument("--heldout-size", type=int, default=60)
    parser.add_argument("--output-file", default="evaluation/extended_eval_prompts.json")
    args = parser.parse_args()

    heldout_records = load_json(Path(args.heldout_file))
    typical = load_json(Path(args.typical_file))
    edge = load_json(Path(args.edge_file))

    ranked = sorted(heldout_records, key=lambda item: overall_score(item["output"]), reverse=True)
    heldout = [
        {
            "id": item["id"],
            "prompt": item["instruction"],
            "category": "heldout_smile",
            "reference": item["output"],
        }
        for item in ranked[: args.heldout_size]
    ]

    payload = heldout + typical + edge
    save_json(Path(args.output_file), payload)
    print(
        json.dumps(
            {
                "heldout": len(heldout),
                "typical": len(typical),
                "edge": len(edge),
                "total": len(payload),
                "output_file": args.output_file,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
