#!/usr/bin/env python
"""Create the fixed AlpacaEval-style subset (300 prompts, seed 42)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_sets/alpaca_eval_300_seed42.json")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source_json",
        default=None,
        help="Optional local JSON list with an instruction/prompt field.",
    )
    return parser.parse_args()


def _rows_from_hf() -> list[dict]:
    """Load a lightweight instruction set that does not rely on deprecated dataset scripts."""
    from datasets import load_dataset

    # yahma/alpaca-cleaned is parquet-backed and widely available.
    raw = load_dataset("yahma/alpaca-cleaned", split="train")
    rows = []
    for row in raw:
        instruction = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        if inp:
            instruction = f"{instruction}\n\n{inp}".strip()
        if instruction:
            rows.append({"instruction": instruction, "dataset": "alpaca_eval"})
    return rows


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.source_json:
        raw = json.loads(Path(args.source_json).read_text(encoding="utf-8"))
        rows = []
        for row in raw:
            instruction = (row.get("instruction") or row.get("prompt") or "").strip()
            if instruction:
                rows.append({"instruction": instruction, "dataset": "alpaca_eval"})
    else:
        rows = _rows_from_hf()

    if len(rows) < args.n:
        raise RuntimeError(f"Only found {len(rows)} prompts; need {args.n}")

    random.shuffle(rows)
    subset = rows[: args.n]
    for i, row in enumerate(subset):
        row["id"] = f"alpaca-eval-42-{i:04d}"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(subset)} prompts to {out}")


if __name__ == "__main__":
    main()
