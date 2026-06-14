"""Presence and marker checks for the validation procedure docs and CONTEXT updates."""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    """Walk up to the directory holding docs/validation/pil-procedure.md (or its parent docs/)."""
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


def test_sim_context_mentions_matrix_and_seam() -> None:
    """sim CONTEXT.md documents the config matrix and the step_once seam."""
    text = _read("packages/sim/src/sim/CONTEXT.md")
    assert "step_once" in text
    assert "EnvironmentConfig" in text


def test_sim_context_cites_canonical_build_tc_packet_home() -> None:
    """sim CONTEXT.md cites flight.libs.commands as the canonical build_tc_packet import."""
    text = _read("packages/sim/src/sim/CONTEXT.md")
    assert "flight.libs.commands" in text
    assert "build_tc_packet" in text


def test_gse_context_present_with_permanent_gap() -> None:
    """gse CONTEXT.md exists and records the permanent ground-segment gap."""
    text = _read("packages/gse/src/gse/CONTEXT.md")
    assert "ground segment" in text.lower()
    assert "step_once" in text
