#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


CORE_KEYS = ["distinct_1", "distinct_2", "avg_length", "negative_behavior_frequency"]
EMPATHY_KEYS = ["comfort", "understanding", "encouragement", "exploration"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_core(metrics: dict[str, dict], output_dir: Path) -> None:
    labels = list(metrics.keys())
    for key in CORE_KEYS:
        values = [metrics[name].get(key, 0.0) for name in labels]
        plt.figure(figsize=(8, 4.5))
        plt.bar(labels, values)
        plt.title(key)
        plt.tight_layout()
        plt.savefig(output_dir / f"{key}.png", dpi=200)
        plt.close()


def plot_empathy(metrics: dict[str, dict], output_dir: Path) -> None:
    labels = list(metrics.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for idx, key in enumerate(EMPATHY_KEYS):
        values = [metrics[name].get("empathy_behavior_proxy", {}).get(key, 0.0) for name in labels]
        axes[idx].bar(labels, values)
        axes[idx].set_title(key)
    fig.tight_layout()
    fig.savefig(output_dir / "empathy_proxy.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    metrics = {}
    for item in args.metric_files:
        name, path = item.split("=", 1)
        metrics[name] = load_json(Path(path))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_core(metrics, output_dir)
    plot_empathy(metrics, output_dir)
    (output_dir / "metric_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"models": list(metrics.keys()), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
