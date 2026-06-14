"""Tests for the scenario runner + suite (runner, characterize)."""

import pandas as pd
from tools.analysis.characterize import (
    repo_root,
    run_suite,
    scenario_file_paths,
    suite_specs,
)
from tools.analysis.runner import (
    SCENARIOS,
    ScenarioRun,
    load_scenario_spec,
    run_scenario,
    scenario,
    scenario_names,
)


def _final(run: ScenarioRun, group: str, column: str) -> object:
    """Return the final value of a wide-frame column for a run."""
    return run.capture.wide[group][column].iloc[-1]


def _final_float(run: ScenarioRun, group: str, column: str) -> float:
    """Return the final value of a numeric wide-frame column as a float."""
    return float(run.capture.wide[group][column].iloc[-1])


def _ever_positive(run: ScenarioRun, group: str, column: str) -> bool:
    """Return True if a wide-frame column was ever positive over the run."""
    return bool((pd.to_numeric(run.capture.wide[group][column]) > 0).any())


def test_every_builtin_scenario_runs() -> None:
    """Every built-in scenario runs and captures the requested number of steps."""
    for name in scenario_names():
        run = run_scenario(SCENARIOS[name])
        assert run.capture.n_steps == SCENARIOS[name].steps


def test_thermal_and_power_drive_safe_and_stow() -> None:
    """Thermal/power over-limit runs latch SAFE and stow the gimbal."""
    for name in ("thermal_over_limit_safe", "power_over_limit_safe"):
        run = run_scenario(scenario(name))
        assert _ever_positive(run, "system", "system.safe_latched")
        assert _ever_positive(run, "payload", "payload.stow_switch")


def test_injected_faults_drive_safe() -> None:
    """The gimbal-runaway and watchdog runs reach SAFE via the injected fault."""
    for name in ("gimbal_runaway", "watchdog_process_died"):
        run = run_scenario(scenario(name))
        assert _final(run, "system", "system.safe_latched") == 1.0


def test_exit_safe_recovery_unlatches() -> None:
    """The recovery run latches SAFE but ends un-latched and back in operations."""
    run = run_scenario(scenario("exit_safe_recovery"))
    assert _ever_positive(run, "system", "system.safe_latched")
    assert _final(run, "system", "system.safe_latched") == 0.0
    assert _final(run, "payload", "payload.gimbal_state") != "SAFE"


def test_model_lifecycle_activates_then_rolls_back() -> None:
    """The model run activates a valid model then rolls back the invalid one."""
    run = run_scenario(scenario("model_lifecycle"))
    states = set(run.capture.wide["model_deploy"]["model_deploy.state"])
    assert "STAGED" in states
    assert _final(run, "model_deploy", "model_deploy.state") == "ROLLBACK_AVAILABLE"


def test_storage_eviction_drops_entries() -> None:
    """The storage run evicts entries once the shrunk quota is exceeded."""
    run = run_scenario(scenario("storage_eviction"))
    assert _final_float(run, "storage", "storage.dropped_count") > 0.0


def test_downlink_backs_up_during_los() -> None:
    """The downlink run accumulates a queue backlog while the link is LOS."""
    run = run_scenario(scenario("downlink_aos_budget"))
    assert _ever_positive(run, "downlink", "downlink.pending_items")


def test_launch_lock_interlock_inhibits_motion() -> None:
    """With the lock engaged the payload inhibits gimbal motion and the lock stays engaged."""
    run = run_scenario(scenario("launch_lock_interlock"))
    assert _ever_positive(run, "payload", "payload.motion_inhibited")
    assert _final(run, "mechanical", "mechanical.launch_lock_state") == "ENGAGED"


def test_command_ingress_routes_and_acks() -> None:
    """A signed command is published by ingress, routed, and acked."""
    run = run_scenario(scenario("command_ingress_auth"))
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


def test_full_suite_includes_builtins_and_files() -> None:
    """The full suite is every built-in scenario plus every scenario file."""
    specs = suite_specs("full")
    assert len(specs) == len(scenario_names()) + len(scenario_file_paths())


def test_run_suite_smoke() -> None:
    """The smoke suite captures exactly the nominal scenario."""
    runs = run_suite("smoke")
    assert len(runs) == 1
    assert runs[0].spec.name == "nominal_tracking"


def test_repo_root_locates_scenarios() -> None:
    """The repo root resolver finds the scenarios directory."""
    assert (repo_root() / "scenarios").is_dir()
