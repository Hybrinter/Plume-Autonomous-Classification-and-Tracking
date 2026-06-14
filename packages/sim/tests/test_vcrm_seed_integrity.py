"""Integrity checks on the seeded VCRM source of truth.

Asserts vcrm.toml parses, only seeds RUNNING-venue requirements (plus the permanent gap),
and that every cited Satisfies: REQ-ID actually appears in a flight module docstring.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    """Walk up from this test file to the directory holding docs/requirements/vcrm.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "docs" / "requirements" / "vcrm.toml").exists():
            return parent
    raise FileNotFoundError("could not locate docs/requirements/vcrm.toml above the test file")


_RUNNING_VENUES = {"unit", "sil", "sil-link-real"}
_ALL_VENUES = _RUNNING_VENUES | {"pil", "hil", "none"}
_METHODS = {"unit", "SIL", "PIL", "HIL", "none"}


def _vcrm_path() -> Path:
    """Absolute path to the seeded vcrm.toml."""
    return _repo_root() / "docs" / "requirements" / "vcrm.toml"


def _flight_src() -> Path:
    """Absolute path to the flight source tree."""
    return _repo_root() / "packages" / "flight" / "src"


def _load() -> dict[str, Any]:
    """Parse vcrm.toml into a dict."""
    with _vcrm_path().open("rb") as handle:
        return tomllib.load(handle)


def _cited_req_ids() -> set[str]:
    """Collect every REQ-ID appearing after a 'Satisfies:' marker in flight sources."""
    found: set[str] = set()
    for path in _flight_src().rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "Satisfies:" not in line:
                continue
            tail = line.split("Satisfies:", 1)[1]
            for token in tail.replace(",", " ").split():
                if token.startswith("REQ-"):
                    found.add(token.strip(".() "))
    return found


def test_vcrm_parses_and_has_requirements() -> None:
    """vcrm.toml must parse and contain at least one requirement."""
    data = _load()
    assert isinstance(data.get("requirement"), list)
    assert len(data["requirement"]) >= 6


def test_every_requirement_has_required_fields() -> None:
    """Each requirement entry carries the full schema with valid enum values."""
    for req in _load()["requirement"]:
        for key in ("id", "statement", "method", "venue", "modules", "evidence", "status"):
            assert key in req, f"{req.get('id')} missing {key}"
        assert req["method"] in _METHODS
        assert req["venue"] in _ALL_VENUES
        assert isinstance(req["modules"], list)
        assert isinstance(req["evidence"], list)


def test_only_running_venues_or_permanent_gap() -> None:
    """Seeded requirements target a running venue, or are the recorded permanent gap."""
    for req in _load()["requirement"]:
        if req["status"] == "gap":
            assert req["venue"] == "none"
        else:
            assert req["venue"] in _RUNNING_VENUES


def test_cited_modules_actually_exist_in_source() -> None:
    """Every REQ-ID listed in a requirement's modules is cited by a flight module."""
    cited = _cited_req_ids()
    for req in _load()["requirement"]:
        for req_id in req["modules"]:
            assert req_id in cited, f"{req_id} not cited by any flight Satisfies: header"


def test_permanent_ground_segment_gap_present() -> None:
    """The permanent 'real ground segment never tested' gap row must exist."""
    gaps = [r for r in _load()["requirement"] if r["status"] == "gap"]
    assert any("ground segment" in r["statement"].lower() for r in gaps)
