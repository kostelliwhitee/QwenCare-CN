#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from training.quality_rules import is_junk_response, quality_breakdown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-file", action="append", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--min-margin", type=float, default=2.0)
    parser.add_argument("--min-usable-candidates", type=int, default=3)
    args = parser.parse_args()

    grouped = {}
    for candidate_file in args.candidate_file:
        payload = json.loads(Path(candidate_file).read_text(encoding="utf-8"))
        for item in payload:
            key = item["id"]
            row = grouped.setdefault(
                key,
                {
                    "id": item["id"],
                    "prompt": item.get("prompt") or item.get("instruction", ""),
                    "candidates": [],
                },
            )
            if item.get("output"):
                row["candidates"].append(item["output"])
            row["candidates"].extend(item.get("candidates", []))

    dpo_pairs = []

    for item in grouped.values():
        usable_candidates = []
        for candidate in item["candidates"]:
            candidate = candidate.strip()
            if is_junk_response(candidate):
                continue
            if candidate in usable_candidates:
                continue
            usable_candidates.append(candidate)
        if len(usable_candidates) < args.min_usable_candidates:
            continue
        ranked = sorted(
            ((candidate, quality_breakdown(candidate)) for candidate in usable_candidates),
            key=lambda pair: pair[1]["total"],
            reverse=True,
        )
        chosen, chosen_meta = ranked[0]
        rejected, rejected_meta = ranked[-1]
        if chosen == rejected:
            continue
        margin = chosen_meta["total"] - rejected_meta["total"]
        if margin < args.min_margin:
            continue
        dpo_pairs.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "meta": {
                    "chosen_score": chosen_meta["total"],
                    "rejected_score": rejected_meta["total"],
                    "margin": margin,
                    "usable_candidates": len(usable_candidates),
                    "chosen_breakdown": chosen_meta,
                    "rejected_breakdown": rejected_meta,
                },
            }
        )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dpo_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pairs": len(dpo_pairs), "output": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
