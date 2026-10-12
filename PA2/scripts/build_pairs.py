#!/usr/bin/env python
"""Build blinded pairwise comparison files for DPO-vs-SFT judging."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_path", required=True)
    parser.add_argument("--dpo_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--sft_name", default="sft")
    parser.add_argument("--dpo_name", default="dpo")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    sft = {row["id"]: row for row in json.loads(Path(args.sft_path).read_text(encoding="utf-8"))}
    dpo = {row["id"]: row for row in json.loads(Path(args.dpo_path).read_text(encoding="utf-8"))}
    shared = sorted(set(sft) & set(dpo))

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for example_id in shared:
            sft_row = sft[example_id]
            dpo_row = dpo[example_id]
            order = [("A", args.sft_name, sft_row["output"]), ("B", args.dpo_name, dpo_row["output"])]
            random.shuffle(order)
            mapping = {label: model for label, model, _ in order}
            responses = {label: text for label, _, text in order}
            record = {
                "id": example_id,
                "instruction": sft_row.get("instruction") or sft_row.get("prompt"),
                "response_a": responses["A"],
                "response_b": responses["B"],
                "hidden_mapping": mapping,
                "dataset": sft_row.get("dataset"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(shared)} pairs to {out}")


if __name__ == "__main__":
    main()
