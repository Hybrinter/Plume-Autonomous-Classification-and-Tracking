#!/usr/bin/env python3
"""Check that descriptive docs mirror package source trees.

Exit codes:
  0 — no problems (or warn-only mode with findings printed)
  1 — problems found in strict mode

Usage:
  uv run python scripts/check_docs.py           # warn mode (default)
  uv run python scripts/check_docs.py --strict  # fail on findings
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PACKAGES: dict[str, Path] = {
    "flight": REPO / "packages/flight/src/flight",
    "sim": REPO / "packages/sim/src/sim",
    "gse": REPO / "packages/gse/src/gse",
    "tools": REPO / "packages/tools/src/tools",
}

REQUIRED_MODULE_SECTIONS = (
    "## Purpose",
    "## Public interface",
    "## Inputs and outputs",
    "## Behavior",
    "## Errors and faults",
    "## Messages",
    "## Configuration",
    "## Constraints",
    "## Related documents",
)

REQUIRED_DIR_SECTIONS = (
    "## Purpose",
    "## Contents",
    "## Package interface",
    "## Interactions",
    "## Constraints",
    "## Related documents",
)

ADR_REF_RE = re.compile(r"ADR[- ]?(?:REPO|FLIGHT|SIM|GSE|TOOLS)?-?\d{4}|docs/.*/adr/", re.I)
RATIONALE_RE = re.compile(
    r"\b(because|in order to|rather than|instead of|we chose|the reason|"
    r"so that|designed to|intended to)\b",
    re.I,
)

# Paths allowed to mention ADRs (indexes and records).
ADR_ALLOWED_PREFIXES = (
    REPO / "docs/adr",
    REPO / "docs/flight/adr",
    REPO / "docs/sim/adr",
    REPO / "docs/gse/adr",
    REPO / "docs/tools/adr",
)


def _is_adr_path(path: Path) -> bool:
    """Return True when path is inside an ADR tree or is docs/adr.md / docs/<pkg>/adr.md."""
    if path.name == "adr.md" and path.parent in {
        REPO / "docs",
        REPO / "docs/flight",
        REPO / "docs/sim",
        REPO / "docs/gse",
        REPO / "docs/tools",
    }:
        return True
    return any(path == p or p in path.parents for p in ADR_ALLOWED_PREFIXES)


def expected_docs_for_source(pkg: str, src_root: Path) -> tuple[set[Path], set[Path]]:
    """Return (expected_md_files, expected_dirs) under docs/<pkg>/ plus docs/<pkg>.md."""
    docs_pkg = REPO / "docs" / pkg
    md_files: set[Path] = {REPO / "docs" / f"{pkg}.md"}
    dirs: set[Path] = {docs_pkg}

    for path in src_root.rglob("*"):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.name in {"CONTEXT.md", "py.typed"}:
            continue
        rel = path.relative_to(src_root)
        if path.is_dir():
            parts = rel.parts
            md = (
                docs_pkg.joinpath(*parts[:-1], f"{parts[-1]}.md")
                if len(parts) > 1
                else docs_pkg / f"{parts[0]}.md"
            )
            folder = docs_pkg.joinpath(*parts)
            md_files.add(md)
            dirs.add(folder)
        elif path.suffix == ".py" and path.name != "__init__.py":
            md_files.add(docs_pkg / rel.with_suffix(".md"))
    return md_files, dirs


def iter_descriptive_pages() -> list[Path]:
    """Yield descriptive markdown pages (exclude style, requirements, validation, ADRs)."""
    pages: list[Path] = []
    for pkg in PACKAGES:
        root = REPO / "docs" / f"{pkg}.md"
        if root.exists():
            pages.append(root)
        pkg_dir = REPO / "docs" / pkg
        if not pkg_dir.exists():
            continue
        for path in pkg_dir.rglob("*.md"):
            if _is_adr_path(path):
                continue
            pages.append(path)
    return pages


def check_mirror() -> list[str]:
    """Check source↔docs mirror completeness."""
    findings: list[str] = []
    for pkg, src_root in PACKAGES.items():
        if not src_root.exists():
            findings.append(f"missing source root: {src_root}")
            continue
        expected_md, expected_dirs = expected_docs_for_source(pkg, src_root)
        for md in sorted(expected_md):
            if not md.is_file():
                findings.append(f"missing doc page: {md.relative_to(REPO)}")
        for d in sorted(expected_dirs):
            if not d.is_dir():
                findings.append(f"missing doc directory: {d.relative_to(REPO)}")

        docs_pkg = REPO / "docs" / pkg
        if docs_pkg.exists():
            for path in docs_pkg.rglob("*.md"):
                if _is_adr_path(path):
                    continue
                if path not in expected_md:
                    # Allow only expected mirror pages under package tree.
                    findings.append(f"unexpected doc page: {path.relative_to(REPO)}")
    return findings


def check_sections() -> list[str]:
    """Check required section headers on descriptive pages."""
    findings: list[str] = []
    for page in iter_descriptive_pages():
        text = page.read_text(encoding="utf-8")
        # Directory pages live beside a same-stem directory, or are docs/<pkg>.md.
        stem_dir = page.with_suffix("")
        is_dir_page = (page.parent == REPO / "docs" and page.stem in PACKAGES) or (
            stem_dir.is_dir()
        )

        required = REQUIRED_DIR_SECTIONS if is_dir_page else REQUIRED_MODULE_SECTIONS
        for section in required:
            if section not in text:
                findings.append(f"{page.relative_to(REPO)}: missing {section}")
    return findings


def check_banned_language() -> list[str]:
    """Flag rationale words and ADR references on descriptive pages."""
    findings: list[str] = []
    for page in iter_descriptive_pages():
        text = page.read_text(encoding="utf-8")
        if page.name.endswith(".md") and "**Status:** stub" in text:
            continue  # scaffold stubs may still say TODO
        for match in ADR_REF_RE.finditer(text):
            findings.append(f"{page.relative_to(REPO)}: ADR reference {match.group(0)!r}")
        # Skip rationale check on unfinished stubs.
        if "TODO." in text and "**Status:** stub" in text:
            continue
        for match in RATIONALE_RE.finditer(text):
            findings.append(f"{page.relative_to(REPO)}: rationale word {match.group(0)!r}")
    return findings


def main() -> int:
    """Run all descriptive-doc checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when any finding is present",
    )
    args = parser.parse_args()

    findings = check_mirror() + check_sections() + check_banned_language()
    if findings:
        print(f"check_docs: {len(findings)} finding(s)")
        for item in findings:
            print(f"  - {item}")
        return 1 if args.strict else 0

    print("check_docs: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
