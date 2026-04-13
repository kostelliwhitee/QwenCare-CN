#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

from training.quality_rules import (
    contains_negative_behavior,
    has_think_leak,
    imperative_rate,
    is_junk_response,
    too_short,
)


NEGATIVE_PATTERNS = [
    "别想太多",
    "你应该",
    "赶紧",
    "这没什么",
    "自己想办法",
    "我不能帮你",
]

EMPATHY_PATTERNS = {
    "comfort": ["抱抱你", "辛苦了", "不容易", "先休息一下"],
    "understanding": ["我能理解", "听起来", "似乎", "能感觉到"],
    "encouragement": ["你已经", "慢慢来", "可以试试", "愿意的话"],
    "exploration": ["愿意多说一点", "发生了什么", "最让你难受的是", "你怎么看"],
}


def tokenize_chars(text: str) -> list[str]:
    return [char for char in text if not char.isspace()]


def distinct_n(texts: list[str], n: int) -> float:
    total = 0
    uniq = set()
    for text in texts:
        tokens = tokenize_chars(text)
        grams = [tuple(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 0))]
        uniq.update(grams)
        total += len(grams)
    return len(uniq) / total if total else 0.0


def mean_length(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return sum(len(tokenize_chars(text)) for text in texts) / len(texts)


def negative_frequency(texts: list[str]) -> float:
    if not texts:
        return 0.0
    count = sum(contains_negative_behavior(text) for text in texts)
    return count / len(texts)


def empathy_behavior_distribution(texts: list[str]) -> dict[str, float]:
    counter = Counter()
    for text in texts:
        for label, patterns in EMPATHY_PATTERNS.items():
            if any(pattern in text for pattern in patterns):
                counter[label] += 1
    total = len(texts) or 1
    return {label: counter[label] / total for label in EMPATHY_PATTERNS}


def rate_mean(texts: list[str], fn) -> float:
    if not texts:
        return 0.0
    return sum(float(fn(text)) for text in texts) / len(texts)


def load_predictions(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        texts = []
        for item in payload:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                texts.append(item.get("response") or item.get("output") or "")
        return [text for text in texts if text]
    raise ValueError("prediction file must be a JSON list")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    predictions = load_predictions(Path(args.prediction_file))
    metrics = {
        "num_predictions": len(predictions),
        "distinct_1": distinct_n(predictions, 1),
        "distinct_2": distinct_n(predictions, 2),
        "avg_length": mean_length(predictions),
        "negative_behavior_frequency": negative_frequency(predictions),
        "junk_rate": rate_mean(predictions, is_junk_response),
        "think_leak_rate": rate_mean(predictions, has_think_leak),
        "imperative_rate": rate_mean(predictions, imperative_rate),
        "too_short_rate": rate_mean(predictions, too_short),
        "empathy_behavior_proxy": empathy_behavior_distribution(predictions),
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
