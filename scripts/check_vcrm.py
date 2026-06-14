#!/usr/bin/env python3
"""VCRM traceability CI check.

Satisfies: REQ-OPER-HIGH-002 (verifiable, type-safe operational config and traceability).

Parses docs/requirements/vcrm.toml (stdlib tomllib) and enforces two invariants:
  1. Every requirement whose venue is a RUNNING profile (unit | sil | sil-link-real) is cited by
     at least one module docstring ("Satisfies: <ID>") under the flight source tree AND has at
     least one evidence entry.
  2. No requirement whose venue is a non-running profile (pil | hil) claims status="verified".

Exits 0 when both invariants hold; otherwise prints each violation and exits 1. Stdlib only so it
runs in CI without the uv workspace installed.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

_RUNNING_VENUES = frozenset({"unit", "sil", "sil-link-real"})
_NON_RUNNING_VENUES = frozenset({"pil", "hil"})


def _collect_cited_ids(src_root: Path) -> set[str]:
    """Return every REQ-ID following a 'Satisfies:' marker under src_root.

    Args:
        src_root: directory tree to scan for *.py files.

    Returns:
        Set of REQ-ID strings (tokens beginning with 'REQ-').
    """
    cited: set[str] = set()
    for path in src_root.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "Satisfies:" not in line:
                continue
            tail = line.split("Satisfies:", 1)[1]
            for token in tail.replace(",", " ").split():
                stripped = token.strip(".() ")
                if stripped.startswith("REQ-"):
                    cited.add(stripped)
    return cited


def _check(vcrm_path: Path, src_root: Path) -> list[str]:
    """Validate the VCRM and return a list of human-readable violation strings.

    Args:
        vcrm_path: path to vcrm.toml.
        src_root: flight source tree to scan for Satisfies: citations.

    Returns:
        List of violation messages; empty list means the VCRM is consistent.
    """
    with vcrm_path.open("rb") as handle:
        data = tomllib.load(handle)
    cited = _collect_cited_ids(src_root)
    violations: list[str] = []
    for req in data.get("requirement", []):
        req_id = req.get("id", "<missing-id>")
        venue = req.get("venue", "none")
        status = req.get("status", "gap")
        if venue in _RUNNING_VENUES:
            modules = req.get("modules", [])
            if not modules:
                violations.append(f"{req_id}: running venue '{venue}' but no modules listed")
            for module_id in modules:
                if module_id not in cited:
                    violations.append(
                        f"{req_id}: cites module {module_id} not found in any "
                        f"'Satisfies:' header under {src_root}"
                    )
            if not req.get("evidence", []):
                violations.append(f"{req_id}: running venue '{venue}' but no evidence listed")
        if venue in _NON_RUNNING_VENUES and status == "verified":
            violations.append(
                f"{req_id}: venue '{venue}' is not a running profile but status='verified'"
            )
    return violations


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the VCRM check, and return a process exit code.

    Args:
        argv: optional argument vector (defaults to sys.argv[1:]).

    Returns:
        0 if the VCRM is consistent, 1 otherwise.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Check VCRM traceability invariants.")
    parser.add_argument(
        "--vcrm",
        type=Path,
        default=repo_root / "docs" / "requirements" / "vcrm.toml",
        help="Path to vcrm.toml.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=repo_root / "packages" / "flight" / "src",
        help="Flight source tree to scan for Satisfies: citations.",
    )
    args = parser.parse_args(argv)
    violations = _check(args.vcrm, args.src)
    if violations:
        print("VCRM check FAILED:")
        for line in violations:
            print(f"  - {line}")
        return 1
    print("VCRM check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
