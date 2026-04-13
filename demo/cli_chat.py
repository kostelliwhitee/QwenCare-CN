#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from demo.backend import DEFAULT_SYSTEM_PROMPT, backend_label, chat_once


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--save-history", default="")
    args = parser.parse_args()

    print("中文 AI 心理医生 CLI Demo")
    print("仅用于课程项目与情绪支持研究演示，不构成专业医疗诊断或治疗建议。")
    print(f"当前后端: {backend_label()}")
    print("输入 /reset 清空历史，输入 /quit 退出。")

    history: list[dict] = []
    while True:
        try:
            message = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if not message:
            continue
        if message == "/quit":
            print("已退出。")
            break
        if message == "/reset":
            history = []
            print("历史已清空。")
            continue

        payload = chat_once(
            message=message,
            history=history,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        history = payload["history"]
        print(f"\n助手> {payload['response']}")

    if args.save_history:
        output_path = Path(args.save_history)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"历史已保存到 {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
