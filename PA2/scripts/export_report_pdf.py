#!/usr/bin/env python
"""Export report/PA2_report.md to report/PA2_report.pdf."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "report" / "PA2_report.md"
PDF = ROOT / "report" / "PA2_report.pdf"


def main() -> None:
    if not MD.exists():
        raise SystemExit(f"Missing {MD}")

    # Prefer pandoc if present.
    pandoc = shutil.which("pandoc")
    if pandoc:
        cmd = [pandoc, str(MD), "-o", str(PDF), "--pdf-engine=xelatex"]
        print("$", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            print("Wrote", PDF)
            return
        except Exception as exc:  # noqa: BLE001
            print("pandoc/xelatex failed:", exc)

    # Fallback: markdown -> HTML -> PDF via weasy/xhtml2pdf/reportlab-ish.
    try:
        import markdown
        from xhtml2pdf import pisa

        html = markdown.markdown(MD.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
        styled = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
        <style>
        body {{ font-family: DejaVu Sans, Arial, sans-serif; font-size: 11pt; line-height: 1.35; margin: 24px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
        th, td {{ border: 1px solid #444; padding: 4px 6px; vertical-align: top; }}
        h1,h2,h3 {{ color: #111; }}
        code {{ font-size: 9pt; }}
        </style></head><body>{html}</body></html>"""
        with PDF.open("wb") as out:
            status = pisa.CreatePDF(styled, dest=out)
        if status.err:
            raise RuntimeError(f"pisa errors: {status.err}")
        print("Wrote", PDF, "via xhtml2pdf")
        return
    except Exception as exc:  # noqa: BLE001
        print("xhtml2pdf path failed:", exc)

    # Last resort: install deps and retry xhtml2pdf.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "markdown", "xhtml2pdf"], check=False)
    import markdown
    from xhtml2pdf import pisa

    html = markdown.markdown(MD.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    styled = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
    <style>
    body {{ font-family: DejaVu Sans, Arial, sans-serif; font-size: 11pt; line-height: 1.35; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ border: 1px solid #444; padding: 4px 6px; vertical-align: top; }}
    code {{ font-size: 9pt; }}
    </style></head><body>{html}</body></html>"""
    with PDF.open("wb") as out:
        status = pisa.CreatePDF(styled, dest=out)
    if status.err:
        raise SystemExit(f"Failed to write PDF: {status.err}")
    print("Wrote", PDF)


if __name__ == "__main__":
    main()
