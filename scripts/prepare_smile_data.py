#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import re
from pathlib import Path


SYSTEM_PROMPT = (
    "你是一位中文情绪支持助手。请用温和、尊重、非说教的方式回应来访者，"
    "优先表达理解、共情、陪伴、澄清和支持。避免夸大承诺、命令式指责和专业医疗诊断。"
)

INSTRUCTION_PREFIX = (
    "请基于以下心理支持对话历史，生成下一句咨询师回复。"
    "要求：中文、温和、支持性、非说教式、贴合来访者处境。\n\n对话：\n"
)

REJECT_PATTERNS = [
    r"你太矫情",
    r"活该",
    r"必须",
    r"赶紧",
    r"别想太多",
    r"不许",
    r"闭嘴",
]


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def looks_chinese(text: str, threshold: float = 0.35) -> bool:
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    return chinese_count / max(len(text), 1) >= threshold


def is_good_response(text: str) -> bool:
    text = normalize(text)
    if len(text) < 4 or len(text) > 512:
        return False
    return not any(re.search(pattern, text) for pattern in REJECT_PATTERNS)


def conversation_to_records(conversation: list[dict], source_id: str) -> tuple[list[dict], int]:
    history = []
    records = []
    filtered = 0
    for turn_index, turn in enumerate(conversation):
        role = turn.get("role", "").strip().lower()
        content = normalize(turn.get("content", ""))
        if not content:
            continue
        if role == "client":
            history.append(f"来访者：{content}")
            continue
        if role != "counselor":
            continue
        if not history or not looks_chinese(content) or not is_good_response(content):
            filtered += 1
            history.append(f"咨询师：{content}")
            continue
        dialogue_history = "\n".join(history[-8:])
        if not looks_chinese(dialogue_history):
            filtered += 1
            history.append(f"咨询师：{content}")
            continue
        digest = hashlib.sha256(f"{dialogue_history}\n{content}".encode("utf-8")).hexdigest()[:12]
        records.append(
            {
                "id": f"smile-{source_id}-{turn_index}-{digest}",
                "dataset": "smilechat",
                "system": SYSTEM_PROMPT,
                "instruction": INSTRUCTION_PREFIX + dialogue_history + "\n咨询师：",
                "input": "",
                "output": content,
            }
        )
        history.append(f"咨询师：{content}")
    return records, filtered


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_records(records: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    data = records[:]
    rng.shuffle(data)
    total = len(data)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    return {
        "train": data[:train_end],
        "validation": data[train_end:val_end],
        "test": data[val_end:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no json files found in {input_dir}")

    all_records = []
    filtered_turns = 0
    for file_path in files:
        conversation = json.loads(file_path.read_text(encoding="utf-8"))
        records, filtered = conversation_to_records(conversation, file_path.stem)
        all_records.extend(records)
        filtered_turns += filtered

    dedup = {}
    for item in all_records:
        key = (item["instruction"], item["output"])
        dedup[key] = item
    final_records = list(dedup.values())
    splits = split_records(final_records, args.seed)

    output_dir = Path(args.output_dir)
    for split, payload in splits.items():
        save_json(output_dir / f"{split}.json", payload)

    quality_report = {
        "source": "SmileChat",
        "input_dir": str(input_dir),
        "num_source_files": len(files),
        "num_records_before_dedup": len(all_records),
        "num_records_after_dedup": len(final_records),
        "filtered_turns": filtered_turns,
        "split_sizes": {split: len(items) for split, items in splits.items()},
    }
    save_json(output_dir / "dataset_stats.json", quality_report)
    save_json(Path("data/samples") / "smile_samples.json", {k: v[:5] for k, v in splits.items()})

    print(json.dumps(quality_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
