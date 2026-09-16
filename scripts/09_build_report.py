"""Build reports/report/report.pdf from reports/report/report.md.

The Markdown is converted to HTML (figures and tables are referenced straight from results/),
and a headless Chrome or Edge prints the HTML to PDF, so no LaTeX installation is needed.
Writing notes left in the source as <!-- NOTE ... --> comments never reach the PDF, and any
unfinished [WRITE] placeholder is listed so the report is not submitted half-written.

Usage (from the repository root):
    python scripts/09_build_report.py
    python scripts/09_build_report.py --browser "C:/path/to/chrome.exe"
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

from milan_forecasting import config

REPORT_DIR = config.ROOT / "reports" / "report"
SOURCE = REPORT_DIR / "report.md"
STYLE = REPORT_DIR / "style.css"
HTML_OUT = REPORT_DIR / "report.html"
PDF_OUT = REPORT_DIR / "report.pdf"

BROWSER_CANDIDATES = [
    "chrome", "google-chrome", "chromium", "msedge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_browser(explicit: str | None) -> str:
    for candidate in [explicit] if explicit else BROWSER_CANDIDATES:
        found = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if found:
            return found
    sys.exit("No Chrome or Edge found; pass its path with --browser.")


def to_html(text: str) -> str:
    title = re.search(r"^# (.+)$", text, re.MULTILINE).group(1)
    body = markdown.markdown(text, extensions=["tables", "attr_list", "footnotes", "md_in_html"])
    css = STYLE.read_text(encoding="utf-8")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
            f"<style>{css}</style></head><body>{body}</body></html>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--browser", help="path to a Chrome or Edge executable")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    text = SOURCE.read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    missing = [line.strip() for line in text.splitlines() if "[WRITE" in line]
    HTML_OUT.write_text(to_html(text), encoding="utf-8")

    subprocess.run([
        find_browser(args.browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        "--allow-file-access-from-files", f"--print-to-pdf={PDF_OUT}", HTML_OUT.as_uri(),
    ], check=True, capture_output=True)
    print(f"wrote {PDF_OUT.relative_to(config.ROOT)}")
    if missing:
        print(f"\n{len(missing)} section(s) still to write:")
        for line in missing:
            print("  -", line[:100])


if __name__ == "__main__":
    main()
