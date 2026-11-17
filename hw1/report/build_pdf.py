"""Render report/README.md to report/hw1_report.pdf via headless Chrome.

Usage (from hw1/):
    uv run --with markdown python report/build_pdf.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SRC = HERE / "README.md"
OUT = HERE / "hw1_report.pdf"

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

CSS = """
@page { size: Letter; margin: 0.8in 0.75in; }
body { font-family: "Latin Modern Roman", "CMU Serif", Georgia, "Times New Roman", serif;
       font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 18pt; text-align: center; margin: 0 0 4pt; line-height: 1.25; }
h2 { font-size: 13.5pt; border-bottom: 1px solid #999; padding-bottom: 2pt; margin-top: 18pt; }
h3 { font-size: 11.5pt; margin-top: 14pt; }
h4 { font-size: 10.5pt; margin: 12pt 0 4pt; }
h2, h3, h4 { break-after: avoid; }
p, li { text-align: justify; }
blockquote { border: none; margin: 0 0 10pt; padding: 0; text-align: center;
             color: #444; font-size: 9.5pt; }
blockquote p { text-align: center; margin: 2pt 0; }
table { border-collapse: collapse; margin: 8pt auto; font-size: 9pt; break-inside: avoid; }
th, td { border-top: 1px solid #bbb; border-bottom: 1px solid #bbb; padding: 3pt 7pt;
         vertical-align: top; text-align: left; }
th { border-top: 1.5px solid #000; border-bottom: 1px solid #000; background: #f3f3f3; }
code { font-family: "DejaVu Sans Mono", Menlo, monospace; font-size: 8.5pt;
       background: #f2f2f2; padding: 0 2px; border-radius: 2px; }
img { display: block; max-width: 100%; margin: 6pt auto; break-inside: avoid; }
p:has(> img) { break-inside: avoid; }
hr { border: none; border-top: 1px solid #ddd; margin: 12pt 0; }
a { color: #1a4f8b; text-decoration: none; }
"""


def protect_math(text: str) -> tuple[str, list[str]]:
    """Swap $...$ spans for placeholders so Markdown doesn't mangle `_` and `\\`."""
    spans: list[str] = []

    def repl(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"@@MATH{len(spans) - 1}@@"

    return re.sub(r"\$[^$\n]+\$", repl, text), spans


def main() -> None:
    text, spans = protect_math(SRC.read_text())
    html_body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    for i, s in enumerate(spans):
        html_body = html_body.replace(f"@@MATH{i}@@", s)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script src="{KATEX}/katex.min.js"></script>
<script src="{KATEX}/contrib/auto-render.min.js"></script>
<style>{CSS}</style></head><body>{html_body}
<script>renderMathInElement(document.body,
  {{delimiters: [{{left: "$", right: "$", display: false}}]}});</script>
</body></html>"""

    chrome = next(
        (c for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c)),
        None,
    )
    if chrome is None:
        raise SystemExit("Chrome/Chromium not found")

    # Write the HTML next to README.md so relative image paths (figures/...) resolve.
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=HERE, delete=False) as f:
        f.write(html)
        tmp = Path(f.name)
    try:
        subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--no-pdf-header-footer", "--virtual-time-budget=10000",
             f"--print-to-pdf={OUT}", tmp.as_uri()],
            check=True, capture_output=True,
        )
    finally:
        tmp.unlink()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
