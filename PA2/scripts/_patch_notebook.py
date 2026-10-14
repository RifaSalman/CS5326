"""Patch kaggle/pa2_pipeline.ipynb for LoRA resume."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "kaggle" / "pa2_pipeline.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))

CELL2 = r'''import os, sys, shutil, subprocess, traceback
from pathlib import Path

os.environ["PYTHONPATH"] = str(Path.cwd())
Path("outputs").mkdir(exist_ok=True)

# Restore prior artifacts from an attached Kaggle dataset if present.
for cand in [Path("/kaggle/input/pa2-checkpoints"), Path("/kaggle/input/pa2-sft-outputs")]:
    if cand.exists():
        print("Restoring from", cand, flush=True)
        for src in cand.rglob("*"):
            if src.is_file():
                rel = src.relative_to(cand)
                dst = Path("outputs") / rel
                if rel.parts and rel.parts[0] == "outputs":
                    dst = Path(*rel.parts)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
        break

# Promote final -> best when best is missing (SFT and both DPO tracks).
for track in ["sft", "dpo_hh", "dpo_uf"]:
    best = Path(f"outputs/{track}/best")
    final = Path(f"outputs/{track}/final")
    if final.exists() and not (best / "config.json").exists():
        print(f"Promoting outputs/{track}/final -> outputs/{track}/best", flush=True)
        if best.exists():
            shutil.rmtree(best)
        shutil.copytree(final, best)

def py(*args):
    subprocess.run([sys.executable, *args], check=True)

def should(*names):
    return STAGE in names or STAGE == "all"

def archive():
    shutil.make_archive("/kaggle/working/pa2_outputs", "zip", "outputs")
    print("Wrote /kaggle/working/pa2_outputs.zip", flush=True)

try:
    if should("sft"):
        py("scripts/train_sft.py", "--output_dir", "outputs/sft")
        py("scripts/plot_metrics.py", "--metrics_path", "outputs/sft/metrics.jsonl", "--output_dir", "report/figures/sft", "--title", "SFT")

    if should("dpo_hh", "resume_dpo") and not Path("outputs/dpo_hh/best/config.json").exists():
        py("scripts/train_dpo.py", "--track", "hh", "--sft_checkpoint", "outputs/sft/best", "--output_dir", "outputs/dpo_hh")
        py("scripts/plot_metrics.py", "--metrics_path", "outputs/dpo_hh/metrics.jsonl", "--output_dir", "report/figures/dpo_hh", "--title", "DPO HH")

    if should("dpo_uf", "resume_dpo") and not Path("outputs/dpo_uf/best/config.json").exists():
        py("scripts/train_dpo.py", "--track", "ultrafeedback", "--sft_checkpoint", "outputs/sft/best", "--output_dir", "outputs/dpo_uf")
        py("scripts/plot_metrics.py", "--metrics_path", "outputs/dpo_uf/metrics.jsonl", "--output_dir", "report/figures/dpo_uf", "--title", "DPO UltraFeedback")

    if should("lora", "resume_dpo", "lora_generate") and not (
        Path("outputs/lora_persona_dpo/best/adapter_config.json").exists()
        or Path("outputs/lora_persona_dpo/best/config.json").exists()
    ):
        py("scripts/train_lora_dpo.py", "--output_dir", "outputs/lora_persona_dpo")

    if should("generate", "resume_dpo", "lora_generate"):
        needed = [
            Path("outputs/sft/best/config.json"),
            Path("outputs/dpo_uf/best/config.json"),
            Path("outputs/dpo_hh/best/config.json"),
        ]
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            print("Skipping generate; missing checkpoints:", missing, flush=True)
        else:
            if not Path("eval_sets/alpaca_eval_300_seed42.json").exists():
                py("scripts/prepare_alpaca_eval.py", "--output", "eval_sets/alpaca_eval_300_seed42.json")
            py("scripts/generate.py", "--model_path", "outputs/sft/best", "--prompts_path", "eval_sets/alpaca_eval_300_seed42.json", "--output_path", "outputs/gen/sft_alpaca.json", "--generator_name", "sft")
            py("scripts/generate.py", "--model_path", "outputs/dpo_uf/best", "--prompts_path", "eval_sets/alpaca_eval_300_seed42.json", "--output_path", "outputs/gen/dpo_uf_alpaca.json", "--generator_name", "dpo_uf")
            py("scripts/generate.py", "--model_path", "outputs/sft/best", "--prompts_path", "eval_sets/hh_rlhf_eval_300_seed42.json", "--output_path", "outputs/gen/sft_hh.json", "--generator_name", "sft")
            py("scripts/generate.py", "--model_path", "outputs/dpo_hh/best", "--prompts_path", "eval_sets/hh_rlhf_eval_300_seed42.json", "--output_path", "outputs/gen/dpo_hh.json", "--generator_name", "dpo_hh")
            py("scripts/build_pairs.py", "--sft_path", "outputs/gen/sft_alpaca.json", "--dpo_path", "outputs/gen/dpo_uf_alpaca.json", "--output_path", "outputs/eval/dpo_uf_vs_sft_pairs.jsonl", "--sft_name", "sft", "--dpo_name", "dpo_uf")
            py("scripts/build_pairs.py", "--sft_path", "outputs/gen/sft_hh.json", "--dpo_path", "outputs/gen/dpo_hh.json", "--output_path", "outputs/eval/dpo_hh_vs_sft_pairs.jsonl", "--sft_name", "sft", "--dpo_name", "dpo_hh")

    if should("judge"):
        py("scripts/judge_pairwise_kaggle.py", "--pairs_path", "outputs/eval/dpo_uf_vs_sft_pairs.jsonl", "--output_path", "outputs/eval/dpo_uf_vs_sft_judged.jsonl", "--reuse_judgments")
        py("scripts/judge_pairwise_kaggle.py", "--pairs_path", "outputs/eval/dpo_hh_vs_sft_pairs.jsonl", "--output_path", "outputs/eval/dpo_hh_vs_sft_judged.jsonl", "--reuse_judgments", "--safety_focus")
except Exception:
    traceback.print_exc()
    raise
finally:
    archive()
    print("Done stage", STAGE, flush=True)
'''

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "STAGE =" in src and "REPO_URL" in src:
        src = re.sub(r'STAGE\s*=\s*"[^"]+"', 'STAGE = "lora"', src)
        cell["source"] = [line + "\n" for line in src.splitlines()]
        print("Updated STAGE cell")
    if "Restoring from" in src or 'should("lora"' in src:
        cell["source"] = [line + "\n" for line in CELL2.splitlines()]
        print("Updated pipeline cell")

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "STAGE =" in src and "REPO_URL" in src:
        for line in src.splitlines():
            if "STAGE" in line:
                print("STAGE line:", line)

p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("Wrote", p)
