#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


CORE_FIELDS = [
    "distinct_1",
    "distinct_2",
    "avg_length",
    "negative_behavior_frequency",
    "junk_rate",
    "think_leak_rate",
    "imperative_rate",
    "too_short_rate",
]

SCENARIO_FIELDS = [
    "avg_empathy",
    "avg_supportiveness",
    "avg_exploration",
    "avg_safety",
    "avg_overall",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_row(name: str, result_dir: Path) -> dict:
    metrics = load_json(result_dir / "metrics.json")
    scenario = load_json(result_dir / "scenario_scores.json")
    row = {"model": name}
    for field in CORE_FIELDS:
        row[field] = metrics.get(field)
    for field in SCENARIO_FIELDS:
        row[field] = scenario.get(field)
    return row


def write_markdown(rows: list[dict], output_path: Path) -> None:
    headers = ["model", *CORE_FIELDS, *SCENARIO_FIELDS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for field in headers:
            value = row[field]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True, help="name=path")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = []
    for item in args.results:
        name, path = item.split("=", 1)
        rows.append(build_row(name, Path(path)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = ["model", *CORE_FIELDS, *SCENARIO_FIELDS]
    with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "comparison_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(rows, output_dir / "comparison_summary.md")

    print(json.dumps({"rows": len(rows), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
