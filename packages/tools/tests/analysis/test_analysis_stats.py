"""Tests for the per-signal summary statistics (stats)."""

import pytest
from tools.analysis.runner import ScenarioRun
from tools.analysis.stats import STATS_COLUMNS, summarize

pytestmark = pytest.mark.slow


def test_summarize_has_one_row_per_emitted_column(builtin_runs: dict[str, ScenarioRun]) -> None:
    """The stats frame has exactly one row per emitted column, with the declared columns."""
    run = builtin_runs["nominal_tracking"]
    stats = summarize(run.capture)
    assert stats.shape[0] == run.capture.n_columns
    assert list(stats.columns) == list(STATS_COLUMNS)


def test_numeric_stats_are_sane(builtin_runs: dict[str, ScenarioRun]) -> None:
    """A monotonic numeric signal reports the expected min/max/first/last."""
    run = builtin_runs["nominal_tracking"]
    stats = summarize(run.capture).set_index("signal")
    step = stats.loc["system.step"]
    assert step["kind"] == "NUMERIC"
    assert float(step["min"]) == 1.0
    assert float(step["max"]) == float(run.capture.n_steps)
    assert step["first"] == "1"
    assert step["last"] == str(run.capture.n_steps)


def test_categorical_stats_capture_transitions_and_mode(
    builtin_runs: dict[str, ScenarioRun],
) -> None:
    """A power-SAFE run shows the gimbal FSM transitioning and a non-empty modal state."""
    run = builtin_runs["power_over_limit_safe"]
    stats = summarize(run.capture).set_index("signal")
    fsm = stats.loc["payload.gimbal_state"]
    assert fsm["kind"] == "CATEGORICAL"
    assert int(fsm["n_transitions"]) >= 1
    assert fsm["mode"] != ""
    assert fsm["last"] == "SAFE"


def test_safe_latch_stats_reflect_the_fault(builtin_runs: dict[str, ScenarioRun]) -> None:
    """The SAFE-latch numeric signal rises from 0 to 1 over a SAFE run."""
    run = builtin_runs["power_over_limit_safe"]
    stats = summarize(run.capture).set_index("signal")
    latch = stats.loc["system.safe_latched"]
    assert float(latch["min"]) == 0.0
    assert float(latch["max"]) == 1.0
    assert int(latch["n_transitions"]) == 1
