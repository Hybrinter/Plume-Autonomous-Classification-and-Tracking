#!/usr/bin/env python3
"""Check ADR header schema, indexes, and reference isolation.

Exit codes:
  0 — no problems (or warn-only mode with findings printed)
  1 — problems found in strict mode

Usage:
  uv run python scripts/check_adr.py
  uv run python scripts/check_adr.py --strict
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SCOPES: dict[str, Path] = {
    "REPO": REPO / "docs/adr",
    "FLIGHT": REPO / "docs/flight/adr",
    "SIM": REPO / "docs/sim/adr",
    "GSE": REPO / "docs/gse/adr",
    "TOOLS": REPO / "docs/tools/adr",
}

INDEX_FILES: dict[str, Path] = {
    "REPO": REPO / "docs/adr.md",
    "FLIGHT": REPO / "docs/flight/adr.md",
    "SIM": REPO / "docs/sim/adr.md",
    "GSE": REPO / "docs/gse/adr.md",
    "TOOLS": REPO / "docs/tools/adr.md",
}

HEADER_FIELDS = (
    "Status",
    "Date",
    "Topic",
    "Supersedes",
    "Superseded-by",
    "Related",
)

STATUS_VALUES = {"Proposed", "Accepted", "Superseded", "Deprecated", "Rejected"}
TOPIC_VALUES = {
    "rename",
    "restructure",
    "feature-add",
    "feature-remove",
    "interface",
    "dependency",
    "sim-fidelity",
    "validation",
    "safety",
    "tooling",
}

TITLE_RE = re.compile(r"^# ADR-(REPO|FLIGHT|SIM|GSE|TOOLS)-(\d{4}): .+")
FIELD_RE = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$")
ADR_REF_RE = re.compile(
    r"ADR[- ]?(?:REPO|FLIGHT|SIM|GSE|TOOLS)?-?\d{4}|docs/(?:flight/|sim/|gse/|tools/)?adr/",
    re.I,
)

# Legacy REPO files keep historical titles without ADR-REPO- prefix in the H1.
LEGACY_REPO_FILE_RE = re.compile(r"^(\d{4})-.+\.md$")


def _parse_header(path: Path) -> dict[str, str]:
    """Parse bold header fields from an ADR markdown file."""
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = FIELD_RE.match(line.strip())
        if match:
            fields[match.group(1)] = match.group(2).strip()
        if line.startswith("## "):
            break
    return fields


def check_package_adrs() -> list[str]:
    """Validate new-style package ADR files (if any exist)."""
    findings: list[str] = []
    for scope, folder in SCOPES.items():
        if scope == "REPO":
            continue
        if not folder.exists():
            findings.append(f"missing ADR directory: {folder.relative_to(REPO)}")
            continue
        index = INDEX_FILES[scope]
        if not index.is_file():
            findings.append(f"missing ADR index: {index.relative_to(REPO)}")
        numbers: set[str] = set()
        for path in sorted(folder.glob("*.md")):
            if path.name == "README.md":
                continue
            text = path.read_text(encoding="utf-8")
            first = text.splitlines()[0] if text.splitlines() else ""
            title = TITLE_RE.match(first)
            if not title:
                findings.append(
                    f"{path.relative_to(REPO)}: title must be "
                    f"'# ADR-{scope}-NNNN: ...'"
                )
                continue
            if title.group(1) != scope:
                findings.append(
                    f"{path.relative_to(REPO)}: scope mismatch "
                    f"(got {title.group(1)}, expected {scope})"
                )
            num = title.group(2)
            if num in numbers:
                findings.append(f"{path.relative_to(REPO)}: duplicate number {num}")
            numbers.add(num)
            fields = _parse_header(path)
            for key in HEADER_FIELDS:
                if key not in fields:
                    findings.append(f"{path.relative_to(REPO)}: missing **{key}:**")
            status = fields.get("Status", "")
            if status and status.split()[0] not in STATUS_VALUES:
                # Allow "Accepted (date)" style by taking first token.
                token = status.replace("(", " ").split()[0]
                if token not in STATUS_VALUES:
                    findings.append(
                        f"{path.relative_to(REPO)}: bad Status {status!r}"
                    )
            topic = fields.get("Topic", "")
            if topic and topic not in TOPIC_VALUES:
                findings.append(f"{path.relative_to(REPO)}: bad Topic {topic!r}")
    return findings


def check_repo_legacy() -> list[str]:
    """Ensure legacy REPO ADR decision files still exist."""
    findings: list[str] = []
    folder = SCOPES["REPO"]
    if not folder.is_dir():
        findings.append("missing docs/adr/")
        return findings
    if not INDEX_FILES["REPO"].is_file():
        findings.append("missing docs/adr.md")
    numbered = [
        p for p in folder.glob("*.md") if LEGACY_REPO_FILE_RE.match(p.name)
    ]
    if len(numbered) < 1:
        findings.append("docs/adr/: expected legacy numbered ADR files")
    return findings


def check_adr_refs_outside() -> list[str]:
    """Flag ADR references outside ADR trees (descriptive docs + packages)."""
    findings: list[str] = []
    scan_roots = [
        REPO / "docs",
        REPO / "packages",
        REPO / "CLAUDE.md",
        REPO / "README.md",
    ]
    allowed_dirs = set(SCOPES.values())
    allowed_indexes = set(INDEX_FILES.values())

    def allowed(path: Path) -> bool:
        if path in allowed_indexes:
            return True
        return any(path == d or d in path.parents for d in allowed_dirs)

    files: list[Path] = []
    for root in scan_roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(root.rglob("*.md"))
            files.extend(root.rglob("*.py"))

    for path in files:
        if allowed(path):
            continue
        # Skip archived / style templates that show ADR form.
        if "docs/style" in str(path).replace("\\", "/"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for match in ADR_REF_RE.finditer(text):
            findings.append(
                f"{path.relative_to(REPO)}: ADR reference {match.group(0)!r}"
            )
    return findings


def main() -> int:
    """Run ADR checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when any finding is present",
    )
    parser.add_argument(
        "--skip-refs",
        action="store_true",
        help="skip whole-tree ADR reference scan",
    )
    args = parser.parse_args()

    findings = check_repo_legacy() + check_package_adrs()
    if not args.skip_refs:
        findings.extend(check_adr_refs_outside())

    if findings:
        print(f"check_adr: {len(findings)} finding(s)")
        for item in findings:
            print(f"  - {item}")
        return 1 if args.strict else 0

    print("check_adr: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
