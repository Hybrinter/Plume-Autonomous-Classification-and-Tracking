"""Run the repo-root declarative scenarios through the orchestrator and score them.

Each scenarios/<name>.toml is loaded and driven end-to-end on the real InProcessBackend
over its profile, asserting the full assertion set passes with no failures or skips. Paths
are repo-root-relative (config/default.toml, profiles/, scenarios/) because CI runs pytest
from the repo root; these are the artifacts the VCRM cites as scenario:<name> evidence.
"""

import pytest
from gse.orchestrator import run_scenario
from gse.scenario import load_scenario

# (scenario file stem, expected passing frame-portable assertion count).
_SCENARIOS = [
    ("ingress_auth_accept", 2),
    ("ingress_nack", 2),
    ("closed_loop_pointing", 2),
    ("safe_on_thermal", 1),
    ("command_route_exec", 2),
    ("product_downlink", 2),
]


@pytest.mark.parametrize(("name", "expected_passed"), _SCENARIOS)
def test_declarative_scenario_passes(name: str, expected_passed: int) -> None:
    """The named scenario runs clean: expected passes, zero failures, zero skips."""
    scenario = load_scenario(f"scenarios/{name}.toml")
    report = run_scenario(scenario, f"profiles/{scenario.profile}.toml")

    assert report.failed == 0, [r for r in report.results if r.status == "fail"]
    assert report.skipped == 0, [r for r in report.results if r.status == "skip"]
    assert report.passed == expected_passed, report.results
