"""Set STAGE in kaggle/pa2_pipeline.ipynb."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

stage = sys.argv[1]
p = Path(__file__).resolve().parents[1] / "kaggle" / "pa2_pipeline.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "STAGE =" in src and "REPO_URL" in src:
        src = re.sub(r'STAGE\s*=\s*"[^"]+"', f'STAGE = "{stage}"', src)
        cell["source"] = [line + "\n" for line in src.splitlines()]
        print("set", stage)
p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
