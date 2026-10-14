"""Make judge stage archive only eval outputs (skip multi-GB checkpoints)."""
from __future__ import annotations

import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "kaggle" / "pa2_pipeline.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))

NEW_ARCHIVE = '''def archive():
    # Judge artifacts are small; avoid re-zipping multi-GB checkpoints.
    if STAGE == "judge":
        out = Path("/kaggle/working/pa2_eval")
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree("outputs/eval", out, dirs_exist_ok=True)
        shutil.make_archive("/kaggle/working/pa2_eval", "zip", out)
        print("Wrote /kaggle/working/pa2_eval.zip", flush=True)
    else:
        shutil.make_archive("/kaggle/working/pa2_outputs", "zip", "outputs")
        print("Wrote /kaggle/working/pa2_outputs.zip", flush=True)
'''

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "def archive():" not in src:
        continue
    start = src.find("def archive():")
    end = src.find("\ntry:")
    if end < 0:
        end = src.find("try:")
    src = src[:start] + NEW_ARCHIVE + "\n" + src[end:]
    cell["source"] = [line + "\n" for line in src.splitlines()]
    print("patched archive")

# Ensure STAGE=judge
import re
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "STAGE =" in src and "REPO_URL" in src:
        src = re.sub(r'STAGE\s*=\s*"[^"]+"', 'STAGE = "judge"', src)
        cell["source"] = [line + "\n" for line in src.splitlines()]
        print("STAGE=judge")

p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
