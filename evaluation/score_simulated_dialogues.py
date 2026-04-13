#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

from training.quality_rules import quality_breakdown


def score_text(text: str) -> dict[str, float]:
    breakdown = quality_breakdown(text)
    empathy = breakdown["empathy"] + breakdown["validation"]
    support = breakdown["supportiveness"]
    exploration = breakdown["exploration"]
    penalties = (
        1.5 * breakdown["negative"]
        + 1.0 * breakdown["medicalized"]
        + 1.5 * breakdown["imperative_penalty"]
        + 2.0 * breakdown["too_short"]
        + 2.0 * breakdown["think_leak"]
        + 2.0 * (1.0 - breakdown["usable"])
    )
    total = 1.0 + empathy + support + exploration - penalties
    return {
        "empathy": min(5.0, 1.0 + empathy),
        "supportiveness": min(5.0, 1.0 + support),
        "exploration": min(5.0, 1.0 + exploration),
        "safety": max(1.0, 5.0 - penalties),
        "overall": max(1.0, min(5.0, total)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    predictions = json.loads(Path(args.prediction_file).read_text(encoding="utf-8"))
    scored = []
    for item in predictions:
        response = item["response"]
        score = score_text(response)
        scored.append({**item, "score": score})

    summary = {
        "num_cases": len(scored),
        "avg_empathy": sum(item["score"]["empathy"] for item in scored) / max(len(scored), 1),
        "avg_supportiveness": sum(item["score"]["supportiveness"] for item in scored) / max(len(scored), 1),
        "avg_exploration": sum(item["score"]["exploration"] for item in scored) / max(len(scored), 1),
        "avg_safety": sum(item["score"]["safety"] for item in scored) / max(len(scored), 1),
        "avg_overall": sum(item["score"]["overall"] for item in scored) / max(len(scored), 1),
        "details": scored,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
