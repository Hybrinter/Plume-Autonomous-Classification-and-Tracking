"""Static checks that the GSE isolation import contracts are declared.

Guards that flight and sim cannot import the gse package, enforced by import-linter.
"""

from __future__ import annotations

import configparser
from pathlib import Path


def _importlinter_path() -> Path:
    """Locate the repo-root .importlinter by walking up until it is found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".importlinter"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not locate .importlinter above the test file")


def _load() -> configparser.ConfigParser:
    """Parse the repo-root .importlinter INI file."""
    parser = configparser.ConfigParser()
    parser.read(_importlinter_path(), encoding="utf-8")
    return parser


def test_gse_is_a_root_package() -> None:
    """gse must be registered as a root package so import-linter scans it."""
    parser = _load()
    roots = parser["importlinter"]["root_packages"].split()
    assert "gse" in roots


def test_flight_gse_isolation_contract_declared() -> None:
    """flight must be forbidden from importing gse."""
    section = "importlinter:contract:flight-gse-isolation"
    parser = _load()
    assert parser.has_section(section)
    assert parser[section]["type"].strip() == "forbidden"
    assert parser[section]["source_modules"].split() == ["flight"]
    assert parser[section]["forbidden_modules"].split() == ["gse"]


def test_sim_gse_isolation_contract_declared() -> None:
    """sim must be forbidden from importing gse."""
    section = "importlinter:contract:sim-gse-isolation"
    parser = _load()
    assert parser.has_section(section)
    assert parser[section]["type"].strip() == "forbidden"
    assert parser[section]["source_modules"].split() == ["sim"]
    assert parser[section]["forbidden_modules"].split() == ["gse"]
