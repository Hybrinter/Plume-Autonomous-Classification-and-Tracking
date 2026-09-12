"""Shared fixtures for tools.analysis tests."""

from __future__ import annotations

import pytest
from tools.analysis.runner import SCENARIOS, ScenarioRun, run_scenario, scenario_names


@pytest.fixture(scope="session")
def builtin_runs() -> dict[str, ScenarioRun]:
    """Run every built-in scenario once per session and cache the captures."""
    return {name: run_scenario(SCENARIOS[name]) for name in scenario_names()}
