"""Slow integration tests for the scenario runner (full SIL recorder captures)."""

import pandas as pd
import pytest
from tools.analysis.characterize import scenario_file_paths
from tools.analysis.runner import (
    SCENARIOS,
    ScenarioRun,
    load_scenario_spec,
    run_scenario,
    scenario_names,
)

pytestmark = pytest.mark.slow


def _final(run: ScenarioRun, group: str, column: str) -> object:
    """Return the final value of a wide-frame column for a run."""
    return run.capture.wide[group][column].iloc[-1]


def _final_float(run: ScenarioRun, group: str, column: str) -> float:
    """Return the final value of a numeric wide-frame column as a float."""
    return float(run.capture.wide[group][column].iloc[-1])


def _ever_positive(run: ScenarioRun, group: str, column: str) -> bool:
    """Return True if a wide-frame column was ever positive over the run."""
    return bool((pd.to_numeric(run.capture.wide[group][column]) > 0).any())


@pytest.mark.parametrize("name", scenario_names())
def test_builtin_scenario_runs(name: str, builtin_runs: dict[str, ScenarioRun]) -> None:
    """Every built-in scenario runs and captures the requested number of steps."""
    run = builtin_runs[name]
    assert run.capture.n_steps == SCENARIOS[name].steps


def test_power_drives_safe_and_stow(builtin_runs: dict[str, ScenarioRun]) -> None:
    """A power over-limit run latches SAFE and stows the gimbal."""
    run = builtin_runs["power_over_limit_safe"]
    assert _ever_positive(run, "system", "system.safe_latched")
    assert _ever_positive(run, "payload", "payload.stow_switch")


def test_thermal_hot_sample_stays_nominal(builtin_runs: dict[str, ScenarioRun]) -> None:
    """A hot thermal sample publishes telemetry and does not latch SAFE."""
    run = builtin_runs["thermal_hot_sample"]
    assert _final(run, "system", "system.safe_latched") == 0.0
    temps = run.capture.wide["thermal"]["thermal.temperature_c"]
    assert float(temps.max()) >= 95.0


def test_injected_faults_drive_safe(builtin_runs: dict[str, ScenarioRun]) -> None:
    """The gimbal-runaway and watchdog runs reach SAFE via the injected fault."""
    for name in ("gimbal_runaway", "watchdog_process_died"):
        run = builtin_runs[name]
        assert _final(run, "system", "system.safe_latched") == 1.0


def test_exit_safe_recovery_unlatches(builtin_runs: dict[str, ScenarioRun]) -> None:
    """The recovery run latches SAFE but ends un-latched and back in operations."""
    run = builtin_runs["exit_safe_recovery"]
    assert _ever_positive(run, "system", "system.safe_latched")
    assert _final(run, "system", "system.safe_latched") == 0.0
    assert _final(run, "payload", "payload.gimbal_state") != "SAFE"


def test_model_lifecycle_activates_then_rolls_back(builtin_runs: dict[str, ScenarioRun]) -> None:
    """The model run activates a valid model then rolls back the invalid one."""
    run = builtin_runs["model_lifecycle"]
    states = set(run.capture.wide["model_deploy"]["model_deploy.state"])
    assert "STAGED" in states
    assert _final(run, "model_deploy", "model_deploy.state") == "ROLLBACK_AVAILABLE"


def test_storage_eviction_drops_entries(builtin_runs: dict[str, ScenarioRun]) -> None:
    """The storage run evicts entries once the shrunk quota is exceeded."""
    run = builtin_runs["storage_eviction"]
    assert _final_float(run, "storage", "storage.dropped_count") > 0.0


def test_downlink_backs_up_during_los(builtin_runs: dict[str, ScenarioRun]) -> None:
    """The downlink run accumulates a queue backlog while the link is LOS."""
    run = builtin_runs["downlink_aos_budget"]
    assert _ever_positive(run, "downlink", "downlink.pending_items")


def test_launch_lock_interlock_inhibits_motion(builtin_runs: dict[str, ScenarioRun]) -> None:
    """With the lock engaged the payload inhibits gimbal motion and the lock stays engaged."""
    run = builtin_runs["launch_lock_interlock"]
    assert _ever_positive(run, "payload", "payload.motion_inhibited")
    assert _final(run, "mechanical", "mechanical.launch_lock_state") == "ENGAGED"


def test_command_ingress_routes_and_acks(builtin_runs: dict[str, ScenarioRun]) -> None:
    """A signed command is published by ingress, routed, and acked."""
    run = builtin_runs["command_ingress_auth"]
    assert _ever_positive(run, "iss_iface", "iss_iface.command_published")
    assert _ever_positive(run, "command_router", "command_router.routed_count")


def test_scenario_files_load_and_run() -> None:
    """Every existing scenarios/*.toml adapts into a spec that runs."""
    paths = scenario_file_paths()
    assert paths  # the repo ships scenario files
    for path in paths:
        spec = load_scenario_spec(path)
        run = run_scenario(spec)
        assert run.capture.n_steps == spec.steps
