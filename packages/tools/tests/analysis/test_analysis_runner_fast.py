"""Fast tests for the scenario runner registry (no full SIL captures)."""

from tools.analysis.characterize import repo_root, run_suite, scenario_file_paths, suite_specs
from tools.analysis.runner import scenario_names


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
