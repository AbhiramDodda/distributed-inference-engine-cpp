#!/usr/bin/env python3
"""Read-only artifact checks; human or independent review must assess evidence."""

from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "investigation"
RESUME = Path.home() / "JOB_SEARCH" / "MAIN_DOCUMENTS" / "GENERIC"


def report(label, result, detail):
    print(f"{result}: {label}: {detail}")


def main():
    findings = DOCS / "FINDINGS.md"
    text = findings.read_text(encoding="utf-8") if findings.is_file() else ""
    if not text or "Status: **not started**" in text:
        report("investigation record", "INCOMPLETE", "missing or still marked not started")
    else:
        report("investigation record", "PRESENT", "requires evidence review")

    for name in ("resume_software_engineer.tex", "resume_software_engineer.pdf",
                 "resume_software_engineer_with_IPhO.pdf"):
        path = RESUME / name
        report(name, "PRESENT" if path.is_file() else "MISSING", "existence only")

    pdfinfo = shutil.which("pdfinfo")
    pdf = RESUME / "resume_software_engineer.pdf"
    if pdfinfo and pdf.is_file():
        result = subprocess.run([pdfinfo, str(pdf)], capture_output=True, text=True, check=False)
        pages = next((line.split(":", 1)[1].strip() for line in result.stdout.splitlines()
                      if line.startswith("Pages:")), None)
        report("resume page count", "PASS" if result.returncode == 0 and pages == "1" else "UNVERIFIED",
               f"pages={pages or 'unknown'}; inspect actual PDF")
    else:
        report("resume page count", "UNVERIFIED", "pdfinfo or resume PDF unavailable")

    report("substantive acceptance", "UNVERIFIED",
           "tests, benchmarks, attribution, resume accuracy, cleanup and Git changes require review")
    print("This audit cannot establish completion and does not inspect secrets or alter files.")


if __name__ == "__main__":
    main()
