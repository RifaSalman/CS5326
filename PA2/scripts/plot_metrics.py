#!/usr/bin/env python
"""Plot training metrics from metrics.jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--title", default="Training")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_metrics(Path(args.metrics_path))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = [r for r in rows if r.get("split") == "train" and "loss" in r]
    val = [r for r in rows if r.get("split") == "val" and "loss" in r]

    if train:
        plt.figure(figsize=(7, 4))
        plt.plot([r["step"] for r in train], [r["loss"] for r in train], label="train")
        if val:
            plt.plot([r["step"] for r in val], [r["loss"] for r in val], label="val")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"{args.title} loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "loss.png", dpi=160)
        plt.close()

    if any("reward_accuracy" in r for r in val):
        plt.figure(figsize=(7, 4))
        plt.plot([r["step"] for r in val], [r["reward_accuracy"] for r in val], label="val reward acc")
        if any("reward_margin" in r for r in val):
            plt.plot([r["step"] for r in val], [r["reward_margin"] for r in val], label="val reward margin")
        plt.xlabel("Step")
        plt.ylabel("Metric")
        plt.title(f"{args.title} preference diagnostics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "reward_metrics.png", dpi=160)
        plt.close()

    print(f"Wrote plots to {out}")


if __name__ == "__main__":
    main()
