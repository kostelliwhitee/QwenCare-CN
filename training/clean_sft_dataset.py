#!/usr/bin/env python3
import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

from training.qwen3_no_think import clean_response
from training.quality_rules import overall_score, passes_keep_gate, quality_breakdown


ROLE_PREFIX_PATTERN = re.compile(r"^(?:咨询师|来访者|支持者|用户|助理)[:：]\s*", re.MULTILINE)
TOOL_PATTERN = re.compile(r"</?tool_response>|</?tool_call>", re.IGNORECASE)


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_output(text: str) -> str:
    text = text or ""
    text = TOOL_PATTERN.sub("", text)
    text = ROLE_PREFIX_PATTERN.sub("", text)
    text = clean_response(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    args = parser.parse_args()

    records = load_json(Path(args.input_file))
    cleaned = []
    dropped = 0
    tool_removed = 0
    role_removed = 0
    for record in records:
        updated = deepcopy(record)
        before = updated.get("output", "") or ""
        after = clean_output(before)
        if "<tool_response>" in before or "<tool_call>" in before:
            tool_removed += 1
        if ROLE_PREFIX_PATTERN.search(before):
            role_removed += 1
        if not after or not passes_keep_gate(after):
            dropped += 1
            continue
        updated["output"] = after
        meta = dict(updated.get("meta") or {})
        meta["quality_score"] = overall_score(after)
        meta["quality_breakdown"] = quality_breakdown(after)
        meta["postprocess_cleaned"] = True
        updated["meta"] = meta
        cleaned.append(updated)

    save_json(Path(args.output_file), cleaned)
    save_json(
        Path(args.summary_file),
        {
            "input_file": args.input_file,
            "output_file": args.output_file,
            "num_input_records": len(records),
            "num_output_records": len(cleaned),
            "num_dropped_records": dropped,
            "tool_marker_records": tool_removed,
            "role_prefix_records": role_removed,
        },
    )
    print(
        json.dumps(
            {
                "input": len(records),
                "output": len(cleaned),
                "dropped": dropped,
                "tool_marker_records": tool_removed,
                "role_prefix_records": role_removed,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
