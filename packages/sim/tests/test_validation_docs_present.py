"""Presence and marker checks for validation procedures and package docs."""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    """Walk up to the directory holding docs/validation/ and packages/."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "docs" / "validation").exists() or (parent / "docs").exists():
            if (parent / "packages").exists():
                return parent
    raise FileNotFoundError("could not locate the repo root above the test file")


def _read(rel: str) -> str:
    """Read a repo-relative file as UTF-8 text."""
    return (_repo_root() / rel).read_text(encoding="utf-8")


def test_pil_procedure_defined_not_run() -> None:
    """PIL procedure doc exists and is marked DEFINED, NOT RUN."""
    text = _read("docs/validation/pil-procedure.md")
    assert "DEFINED, NOT RUN" in text
    assert "profiles/pil.toml" in text


def test_hil_procedure_defined_not_run() -> None:
    """HIL procedure doc exists and is marked DEFINED, NOT RUN."""
    text = _read("docs/validation/hil-procedure.md")
    assert "DEFINED, NOT RUN" in text
    assert "profiles/hil.toml" in text


def test_sim_docs_mention_matrix_and_seam() -> None:
    """sim package docs document the config matrix and the step_once seam."""
    text = _read("docs/sim.md")
    assert "step_once" in text
    assert "EnvironmentConfig" in text


def test_sim_docs_cite_canonical_build_tc_packet_home() -> None:
    """sim package docs cite flight.libs.commands as the build_tc_packet home."""
    text = _read("docs/sim.md")
    assert "flight.libs.commands" in text
    assert "build_tc_packet" in text


def test_gse_docs_present_with_permanent_gap() -> None:
    """gse package docs exist and record the permanent ground-segment gap."""
    text = _read("docs/gse.md")
    assert "ground segment" in text.lower()
    assert "step_once" in text
