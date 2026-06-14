"""Drive scripts/check_vcrm.py against the seeded VCRM and malformed fixtures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Walk up to the directory holding scripts/check_vcrm.py."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "scripts" / "check_vcrm.py").exists():
            return parent
    raise FileNotFoundError("could not locate scripts/check_vcrm.py above the test file")


def _script() -> Path:
    """Absolute path to the check_vcrm.py script under test."""
    return _repo_root() / "scripts" / "check_vcrm.py"


def _seeded() -> Path:
    """Absolute path to the real seeded vcrm.toml."""
    return _repo_root() / "docs" / "requirements" / "vcrm.toml"


def _run(vcrm_path: Path, src_root: Path) -> subprocess.CompletedProcess[str]:
    """Invoke check_vcrm.py with explicit --vcrm and --src arguments."""
    return subprocess.run(
        [sys.executable, str(_script()), "--vcrm", str(vcrm_path), "--src", str(src_root)],
        capture_output=True,
        text=True,
    )


def test_seeded_vcrm_passes() -> None:
    """The real seeded vcrm.toml against real flight sources exits 0."""
    result = _run(_seeded(), _repo_root() / "packages" / "flight" / "src")
    assert result.returncode == 0, result.stdout + result.stderr


def test_running_requirement_without_citation_fails(tmp_path: Path) -> None:
    """A running-venue requirement citing an uncited REQ-ID exits nonzero."""
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    (fake_src / "mod.py").write_text('"""Satisfies: REQ-REAL-001."""\n', encoding="utf-8")
    vcrm = tmp_path / "vcrm.toml"
    vcrm.write_text(
        "[[requirement]]\n"
        'id = "REQ-FAKE-999"\n'
        'statement = "uncited"\n'
        'method = "SIL"\n'
        'venue = "sil"\n'
        'modules = ["REQ-FAKE-999"]\n'
        'evidence = ["t"]\n'
        'status = "verified"\n',
        encoding="utf-8",
    )
    result = _run(vcrm, fake_src)
    assert result.returncode != 0
    assert "REQ-FAKE-999" in result.stdout


def test_running_requirement_without_evidence_fails(tmp_path: Path) -> None:
    """A running-venue requirement with empty evidence exits nonzero."""
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    (fake_src / "mod.py").write_text('"""Satisfies: REQ-REAL-001."""\n', encoding="utf-8")
    vcrm = tmp_path / "vcrm.toml"
    vcrm.write_text(
        "[[requirement]]\n"
        'id = "REQ-REAL-001"\n'
        'statement = "cited but no evidence"\n'
        'method = "SIL"\n'
        'venue = "sil"\n'
        'modules = ["REQ-REAL-001"]\n'
        "evidence = []\n"
        'status = "verified"\n',
        encoding="utf-8",
    )
    result = _run(vcrm, fake_src)
    assert result.returncode != 0
    assert "evidence" in result.stdout.lower()


def test_pil_hil_verified_claim_fails(tmp_path: Path) -> None:
    """A pil/hil requirement claiming verified exits nonzero (non-running venue)."""
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    (fake_src / "mod.py").write_text('"""Satisfies: REQ-HW-001."""\n', encoding="utf-8")
    vcrm = tmp_path / "vcrm.toml"
    vcrm.write_text(
        "[[requirement]]\n"
        'id = "REQ-HW-001"\n'
        'statement = "hardware only"\n'
        'method = "HIL"\n'
        'venue = "hil"\n'
        'modules = ["REQ-HW-001"]\n'
        'evidence = ["t"]\n'
        'status = "verified"\n',
        encoding="utf-8",
    )
    result = _run(vcrm, fake_src)
    assert result.returncode != 0
    assert "hil" in result.stdout.lower()
