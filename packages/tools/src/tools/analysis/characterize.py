"""Named scenario suites: resolve a suite (or single scenario) to specs and capture them.

A thin orchestration layer over the runner: it names handy groupings of built-in scenarios, it
discovers the repo's existing scenarios/*.toml files and adapts them into specs, and it runs a
resolved suite, returning one ScenarioRun per scenario. The "full" suite is the default the CLI
drives: every built-in scenario plus every existing scenario file.

Contains:
  - SUITES / suite_names: the named built-in groupings.
  - repo_root / scenario_file_paths: locate the repo and its scenario files.
  - suite_specs / run_suite: resolve a suite (or single name) to specs and capture them.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# stdlib
from pathlib import Path

# internal
from tools.analysis.runner import (
    SCENARIOS,
    ScenarioRun,
    ScenarioSpec,
    load_scenario_spec,
    run_scenario,
    scenario_names,
)

# Named groupings of built-in scenario names (the meta-suites below are resolved specially).
SUITES: dict[str, tuple[str, ...]] = {
    "smoke": ("nominal_tracking",),
    "faults": (
        "thermal_over_limit_safe",
        "power_over_limit_safe",
        "gimbal_runaway",
        "watchdog_process_died",
        "model_lifecycle",
    ),
    "commands": (
        "command_ingress_auth",
        "arm_execute_command",
        "exit_safe_recovery",
    ),
    "resources": (
        "storage_eviction",
        "downlink_aos_budget",
        "launch_lock_interlock",
    ),
}

# Meta-suites resolved without a fixed name list.
_META_SUITES: tuple[str, ...] = ("full", "builtin", "files")


def suite_names() -> tuple[str, ...]:
    """Return every selectable suite name (meta-suites first, then the named groupings)."""
    return (*_META_SUITES, *SUITES)


def repo_root() -> Path:
    """Return the repository root (the nearest ancestor with a scenarios/ dir + pyproject.toml).

    Raises:
        RuntimeError: if no such ancestor exists.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "scenarios").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not locate repo root (no scenarios/ + pyproject.toml ancestor)")


def scenario_file_paths() -> tuple[Path, ...]:
    """Return the repo's existing scenarios/*.toml paths, sorted by name."""
    return tuple(sorted((repo_root() / "scenarios").glob("*.toml")))


def suite_specs(name: str) -> list[ScenarioSpec]:
    """Resolve a suite name (or a single scenario name) to the list of specs to capture.

    Args:
        name: "full" (every built-in + every scenario file), "builtin" (every built-in), "files"
            (every scenario file), a named grouping in SUITES, or a single built-in scenario name.

    Returns:
        The ordered list of ScenarioSpecs to run.

    Raises:
        KeyError: if name is not a known suite or scenario.
    """
    if name == "full":
        return [SCENARIOS[n] for n in scenario_names()] + [
            load_scenario_spec(path) for path in scenario_file_paths()
        ]
    if name == "builtin":
        return [SCENARIOS[n] for n in scenario_names()]
    if name == "files":
        return [load_scenario_spec(path) for path in scenario_file_paths()]
    if name in SUITES:
        return [SCENARIOS[scenario_name] for scenario_name in SUITES[name]]
    if name in SCENARIOS:
        return [SCENARIOS[name]]
    raise KeyError(
        f"unknown suite/scenario: {name!r} (suites {list(suite_names())}; "
        f"scenarios {list(scenario_names())})"
    )


def run_suite(name: str) -> list[ScenarioRun]:
    """Resolve a suite (or single scenario) name and capture every scenario it names."""
    return [run_scenario(spec) for spec in suite_specs(name)]
