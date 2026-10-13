from __future__ import annotations

import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

KERNEL = "salmanrifa/pa2-sft-dpo-pipeline"
OUT = Path("outputs/kaggle_logs_v3.txt")


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    print("status:", api.kernels_status(KERNEL))
    parts: list[str] = []
    n = 0
    for chunk in api.kernels_logs_stream(KERNEL):
        n += 1
        if isinstance(chunk, dict):
            parts.append(str(chunk.get("data") or ""))
        else:
            parts.append(str(chunk))
        if n >= 5000:
            break
    text = "".join(parts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print("events", n, "chars", len(text), "wrote", OUT)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if any(k in line for k in ("Error", "Traceback", "Exception", "stage", "Restoring", "Promoting", "step=", "Done", "FileNotFound", "CUDA", "OOM")):
            print(f"{i}: {line[:350]}")
    print("---- TAIL ----")
    for line in lines[-60:]:
        print(line[:350])


if __name__ == "__main__":
    main()
