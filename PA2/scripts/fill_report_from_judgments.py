#!/usr/bin/env python
"""Fill PA2_report.md tables from judged summary JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "PA2_report.md"


def load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_row(summary: dict, judge_model: str) -> str:
    return "\n".join(
        [
            f"| Judge model | `{judge_model}` |",
            f"| N pairs | {summary.get('n', 0)} |",
            (
                "| DPO wins / SFT wins / ties / unknown / failures | "
                f"{summary.get('win', 0)} / {summary.get('loss', 0)} / "
                f"{summary.get('tie', 0)} / {summary.get('unknown', 0)} / "
                f"{summary.get('failure', 0)} |"
            ),
            f"| Win rate (excl. ties) | {summary.get('win_rate_excluding_ties', 0):.3f} |",
        ]
    )


def infer_judge_model(judged_path: Path, default: str = "Qwen/Qwen2.5-1.5B-Instruct") -> str:
    if not judged_path.exists():
        return default
    with judged_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            return row.get("judge_model") or default
    return default


def pick_examples(judged_path: Path, pairs_path: Path, n: int = 2) -> list[str]:
    pairs = {json.loads(l)["id"]: json.loads(l) for l in pairs_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    wins, losses = [], []
    with judged_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("result") == "win" and len(wins) < n:
                wins.append(row)
            elif row.get("result") == "loss" and len(losses) < n:
                losses.append(row)
    chunks = []
    for label, rows in (("DPO preferred", wins), ("SFT preferred", losses)):
        for row in rows:
            pair = pairs.get(row["id"], {})
            instr = (pair.get("instruction") or "")[:180].replace("\n", " ")
            reason = (row.get("reason") or "")[:220].replace("\n", " ")
            chunks.append(f"- **{label}** (`{row['id']}`): {instr}… — Judge: {reason}")
    return chunks


def main() -> None:
    uf_sum = load_summary(ROOT / "outputs/eval/dpo_uf_vs_sft_judged.summary.json")
    hh_sum = load_summary(ROOT / "outputs/eval/dpo_hh_vs_sft_judged.summary.json")
    uf_model = infer_judge_model(ROOT / "outputs/eval/dpo_uf_vs_sft_judged.jsonl")
    hh_model = infer_judge_model(ROOT / "outputs/eval/dpo_hh_vs_sft_judged.jsonl")

    text = REPORT.read_text(encoding="utf-8")

    uf_block = (
        "### UltraFeedback DPO vs SFT (AlpacaEval-300, seed 42)\n\n"
        "| Metric | Value |\n|--------|-------|\n"
        + fmt_row(uf_sum, uf_model)
    )
    hh_block = (
        "### HH-RLHF DPO vs SFT (synthetic HH eval)\n\n"
        "| Metric | Value |\n|--------|-------|\n"
        + fmt_row(hh_sum, hh_model)
    )

    text = re.sub(
        r"### UltraFeedback DPO vs SFT.*?(?=### HH-RLHF)",
        uf_block + "\n\n",
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r"### HH-RLHF DPO vs SFT.*?(?=### LoRA)",
        hh_block + "\n\n",
        text,
        count=1,
        flags=re.S,
    )

    lora_block = (
        "### LoRA persona (separate style evaluation)\n\n"
        "LoRA-DPO was trained on 220 Ramsay-style preference pairs over SmolLM2-360M-Instruct "
        "(r=16, α=32, 3 epochs). Training converged (`train_loss≈0.47`, `eval_loss≈0.37`). "
        "The adapter is intended to increase blunt, high-agency critique on cooking/coding prompts "
        "while keeping refusals intact on clearly unsafe asks. Qualitative spot-checks should compare "
        "base SmolLM2 vs `outputs/lora_persona_dpo/best` on cooking, debugging, general, and safety prompts.\n"
    )
    text = re.sub(
        r"### LoRA persona \(separate style evaluation\).*?(?=## 6\.)",
        lora_block + "\n",
        text,
        count=1,
        flags=re.S,
    )

    examples = []
    examples += pick_examples(
        ROOT / "outputs/eval/dpo_uf_vs_sft_judged.jsonl",
        ROOT / "outputs/eval/dpo_uf_vs_sft_pairs.jsonl",
    )
    examples += pick_examples(
        ROOT / "outputs/eval/dpo_hh_vs_sft_judged.jsonl",
        ROOT / "outputs/eval/dpo_hh_vs_sft_pairs.jsonl",
    )
    interp = (
        "## 6. Interpretation and Error Analysis\n\n"
        f"UltraFeedback DPO win rate (excl. ties) = **{uf_sum.get('win_rate_excluding_ties', 0):.3f}** "
        f"({uf_sum.get('win', 0)} DPO / {uf_sum.get('loss', 0)} SFT / {uf_sum.get('tie', 0)} ties). "
        f"HH DPO win rate (excl. ties) = **{hh_sum.get('win_rate_excluding_ties', 0):.3f}** "
        f"({hh_sum.get('win', 0)} DPO / {hh_sum.get('loss', 0)} SFT / {hh_sum.get('tie', 0)} ties).\n\n"
        "Example judgments:\n"
        + "\n".join(examples[:6])
        + "\n\n"
        "Caveats: the 0.5B policy is often verbose/repetitive under greedy decoding; the HF judge "
        f"(`{uf_model}`) is weaker than a frontier LLM and can be inconsistent on near-ties; "
        "safety judgments on synthetic HH prompts may conflate style with true refusal quality.\n"
    )
    text = re.sub(r"## 6\. Interpretation and Error Analysis.*?(?=## 7\.)", interp + "\n", text, count=1, flags=re.S)

    REPORT.write_text(text, encoding="utf-8")
    print("Updated", REPORT)


if __name__ == "__main__":
    main()
