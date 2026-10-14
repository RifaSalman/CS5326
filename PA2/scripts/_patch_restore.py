"""Patch notebook restore logic to unzip dir-mode dataset zips."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESTORE = '''# Restore prior artifacts from an attached Kaggle dataset if present.
import zipfile
for cand in [Path("/kaggle/input/pa2-checkpoints"), Path("/kaggle/input/pa2-sft-outputs")]:
    if not cand.exists():
        continue
    print("Restoring from", cand, flush=True)
    # dir-mode zip datasets ship top-level archives like sft.zip / dpo_hh.zip
    for zpath in sorted(cand.glob("*.zip")):
        dest = Path("outputs") / zpath.stem
        print("Extracting", zpath.name, "->", dest, flush=True)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(dest)
    for src in cand.rglob("*"):
        if src.is_file() and src.suffix.lower() != ".zip":
            rel = src.relative_to(cand)
            dst = Path("outputs") / rel
            if rel.parts and rel.parts[0] == "outputs":
                dst = Path(*rel.parts)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
    break
'''

for nb_path in [ROOT / "kaggle" / "pa2_pipeline.ipynb"]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "Restoring from" not in src:
            continue
        # Replace old restore block up to promote section
        start = src.find("# Restore prior artifacts")
        end = src.find("# Promote final")
        if start < 0 or end < 0:
            raise SystemExit(f"markers missing in {nb_path}")
        src = src[:start] + RESTORE + "\n" + src[end:]
        cell["source"] = [line + "\n" for line in src.splitlines()]
        print("patched", nb_path)
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
