"""Fetch a burst of live kernel logs (does not wait forever)."""

from __future__ import annotations

from kaggle.api.kaggle_api_extended import KaggleApi

KERNEL = "salmanrifa/pa2-sft-dpo-pipeline"
MAX_EVENTS = 300


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
        if n >= MAX_EVENTS:
            break
    text = "".join(parts)
    print(f"events={n} chars={len(text)}")
    print("---- TAIL ----")
    print(text[-5000:])


if __name__ == "__main__":
    main()
