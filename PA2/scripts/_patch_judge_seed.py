"""Seed judge pairs from eval_sets/pairs and ensure STAGE=judge."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "kaggle" / "pa2_pipeline.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))

SEED = '''    if should("judge"):
        Path("outputs/eval").mkdir(parents=True, exist_ok=True)
        for name in ["dpo_uf_vs_sft_pairs.jsonl", "dpo_hh_vs_sft_pairs.jsonl"]:
            srcp = Path("eval_sets/pairs") / name
            dstp = Path("outputs/eval") / name
            if srcp.exists() and not dstp.exists():
                shutil.copy2(srcp, dstp)
                print("Seeded", dstp, flush=True)
        py("scripts/judge_pairwise_kaggle.py", "--pairs_path", "outputs/eval/dpo_uf_vs_sft_pairs.jsonl", "--output_path", "outputs/eval/dpo_uf_vs_sft_judged.jsonl", "--reuse_judgments")
        py("scripts/judge_pairwise_kaggle.py", "--pairs_path", "outputs/eval/dpo_hh_vs_sft_pairs.jsonl", "--output_path", "outputs/eval/dpo_hh_vs_sft_judged.jsonl", "--reuse_judgments", "--safety_focus")
'''

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "STAGE =" in src and "REPO_URL" in src:
        src = re.sub(r'STAGE\s*=\s*"[^"]+"', 'STAGE = "judge"', src)
        cell["source"] = [line + "\n" for line in src.splitlines()]
        print("STAGE=judge")
    if 'should("judge")' in src:
        # replace the whole judge block
        start = src.find('    if should("judge"):')
        end = src.find("except Exception:")
        if start < 0 or end < 0:
            raise SystemExit("judge block markers missing")
        src = src[:start] + SEED + "\n" + src[end:]
        cell["source"] = [line + "\n" for line in src.splitlines()]
        print("seeded judge block")

p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", p)
