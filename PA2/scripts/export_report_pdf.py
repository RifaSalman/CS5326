#!/usr/bin/env python
"""Export report/PA2_report.md to report/25280059_Report_PA2.pdf."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
MD = REPORT_DIR / "PA2_report.md"
PDF = REPORT_DIR / "25280059_Report_PA2.pdf"
LEGACY_PDF = REPORT_DIR / "PA2_report.pdf"
LOGO = REPORT_DIR / "lums-university-logo.png"
IMG_CACHE = REPORT_DIR / "_pdf_images"
GITHUB_URL = "https://github.com/RifaSalman/CS5326/tree/main/PA2"


def _ensure_deps() -> None:
    try:
        import markdown  # noqa: F401
        from xhtml2pdf import pisa  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "markdown", "xhtml2pdf", "pillow"],
            check=False,
        )


def _prepare_image(src: Path, max_w: int = 1100) -> Path:
    from PIL import Image

    IMG_CACHE.mkdir(parents=True, exist_ok=True)
    out = IMG_CACHE / (src.stem + f"_{max_w}.jpg")
    img = Image.open(src)
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    if img.width > max_w:
        h = int(img.height * (max_w / img.width))
        img = img.resize((max_w, h))
    img.save(out, format="JPEG", quality=90, optimize=True)
    return out


def _cover_html() -> str:
    if not LOGO.exists():
        raise SystemExit(f"Missing logo: {LOGO}")
    logo = _prepare_image(LOGO, max_w=420)
    return f"""
<div class="cover">
  <img class="logo" src="{logo.as_posix()}" alt="LUMS logo" width="150"/>
  <p class="name">Rifa Salman</p>
  <p class="meta">25280059</p>
  <p class="meta">CS5326 Advanced GenAI and Agents</p>
  <p class="title">Report PA2</p>
  <p class="meta">Github repo:
    <a href="{GITHUB_URL}">{GITHUB_URL}</a>
  </p>
</div>
<div class="page-break"></div>
"""


def md_to_html(text: str) -> str:
    import markdown

    def repl(match: re.Match[str]) -> str:
        alt, src = match.group(1), match.group(2)
        path = Path(src)
        if not path.is_absolute():
            path = (REPORT_DIR / src).resolve()
        if not path.exists():
            return f"<p><em>[missing figure: {src}]</em></p>"
        prepared = _prepare_image(path, max_w=1100)
        return (
            f'<p style="text-align:center;">'
            f'<img src="{prepared.as_posix()}" alt="{alt}" width="480"/>'
            f"<br/><span style='font-size:9pt;'>{alt}</span></p>"
        )

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, text)
    # Autolink bare GitHub URLs that remain in body text.
    text = re.sub(
        r"(?<![\"\(])(https://github\.com/[^\s<]+)",
        r"[\1](\1)",
        text,
    )
    return markdown.markdown(text, extensions=["tables", "fenced_code", "nl2br"])


def link_callback(uri: str, rel: str) -> str:
    path = Path(uri)
    if path.exists():
        return str(path.resolve())
    candidate = (REPORT_DIR / uri).resolve()
    if candidate.exists():
        return str(candidate)
    return uri


def main() -> None:
    if not MD.exists():
        raise SystemExit(f"Missing {MD}")
    _ensure_deps()
    from xhtml2pdf import pisa

    body = md_to_html(MD.read_text(encoding="utf-8"))
    styled = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
@page {{ size: A4; margin: 18mm 16mm 18mm 16mm; }}
body {{
  font-family: Helvetica, Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.4;
  color: #111;
}}
.cover {{
  text-align: center;
  padding-top: 55mm;
}}
.cover .logo {{
  display: block;
  margin: 0 auto 18px auto;
}}
.cover .name {{
  font-size: 16pt;
  font-weight: bold;
  margin: 10px 0 6px 0;
}}
.cover .title {{
  font-size: 14pt;
  font-weight: bold;
  margin: 12px 0 10px 0;
}}
.cover .meta {{
  font-size: 11pt;
  margin: 4px 0;
}}
.cover a {{
  color: #0645AD;
  text-decoration: underline;
}}
.page-break {{
  page-break-after: always;
}}
h2 {{
  font-size: 13pt;
  border-bottom: 1px solid #ccc;
  padding-bottom: 2px;
  margin-top: 1.1em;
  margin-bottom: 0.4em;
}}
h3 {{ font-size: 11.5pt; margin-top: 0.9em; }}
p {{ margin: 0.45em 0; }}
a {{ color: #0645AD; text-decoration: underline; }}
table {{
  width: 100%;
  margin: 10px 0 14px 0;
  font-size: 9pt;
}}
th, td {{
  border: 1px solid #555;
  padding: 4px 6px;
  vertical-align: top;
  text-align: left;
}}
th {{ background-color: #efefef; }}
code {{
  font-family: Courier, monospace;
  font-size: 8.5pt;
}}
img {{
  display: block;
  margin: 6px auto 4px auto;
}}
ul {{ margin: 0.4em 0 0.6em 1.2em; }}
li {{ margin: 0.2em 0; }}
</style></head>
<body>
{_cover_html()}
{body}
</body></html>"""

    with PDF.open("wb") as out:
        status = pisa.CreatePDF(
            styled,
            dest=out,
            encoding="utf-8",
            link_callback=link_callback,
        )
    if status.err:
        raise SystemExit(f"Failed to write PDF: {status.err}")
    if LEGACY_PDF.exists() and LEGACY_PDF.resolve() != PDF.resolve():
        LEGACY_PDF.unlink()
    print("Wrote", PDF, f"({PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
