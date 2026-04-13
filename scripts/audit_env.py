#!/usr/bin/env python3
import importlib.util
import json
import platform
import shutil
import subprocess
import sys


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout.strip() or proc.stderr.strip()
    return proc.returncode, output


def main() -> int:
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executables": {
            "python": shutil.which("python"),
            "python3": shutil.which("python3"),
            "conda": shutil.which("conda"),
            "nvidia_smi": shutil.which("nvidia-smi"),
        },
        "modules": {
            name: has_module(name)
            for name in [
                "torch",
                "transformers",
                "datasets",
                "accelerate",
                "deepspeed",
                "peft",
                "trl",
                "gradio",
                "vllm",
            ]
        },
    }

    if has_module("torch"):
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }

    if shutil.which("nvidia-smi"):
        code, output = run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"])
        report["nvidia_smi_query"] = {"code": code, "output": output}

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
