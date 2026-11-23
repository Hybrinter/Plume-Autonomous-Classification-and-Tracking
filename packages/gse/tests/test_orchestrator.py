"""run_scenario scores frame-portable assertions and skips realtime-only ones."""

from pathlib import Path

import pytest
from flight.libs.types import SystemMode
from gse.harness import TelemetryCapture
from gse.orchestrator import _score_frame_portable, run_scenario
from gse.scenario import Assertion, Scenario, SceneSpec


def _scored_scenario() -> Scenario:
    """All-sim scenario asserting a moved gimbal, an inference floor, and a skipped timing one."""
    return Scenario(
        name="orchestrator-smoke",
        profile="profiles/sil.toml",
        scene=SceneSpec(num_frames=12, seed=0),
        commands=(),
        assertions=(
            Assertion(id="GIMBAL-MOVED", kind="gimbal_moved", value=True, tag="frame-portable"),
            Assertion(id="INF-FLOOR", kind="min_inference_count", value=6, tag="frame-portable"),
            Assertion(
                id="ACK-TIMING",
                kind="ack_within_seconds",
                value=2.0,
                tag="realtime-only",
            ),
        ),
        steps=12,
        dt=1.0,
        initial_mode=SystemMode.OPERATE,
    )


def test_run_scenario_scores_and_skips(deterministic_profiles: dict[str, Path]) -> None:
    """Frame-portable assertions pass; the realtime-only assertion is skipped with a reason."""
    report = run_scenario(_scored_scenario(), str(deterministic_profiles["sil"]))

    assert report.scenario == "orchestrator-smoke"
    assert report.passed == 2
    assert report.failed == 0
    assert report.skipped == 1

    by_id = {r.id: r for r in report.results}
    assert by_id["GIMBAL-MOVED"].status == "pass"
    assert by_id["INF-FLOOR"].status == "pass"
    assert by_id["ACK-TIMING"].status == "skip"
    assert "realtime-only" in by_id["ACK-TIMING"].detail


@pytest.mark.parametrize(
    ("expected", "active", "history", "status"),
    [
        ("IDLE", None, (), "fail"),
        ("SAFE", SystemMode.OPERATE, (SystemMode.SAFE, SystemMode.OPERATE), "fail"),
        ("OPERATE", SystemMode.OPERATE, (SystemMode.SAFE, SystemMode.OPERATE), "pass"),
        ("SAFE", SystemMode.SAFE, (SystemMode.OPERATE, SystemMode.SAFE), "pass"),
        ("IDLE", SystemMode.OPERATE, (), "fail"),
        ("IDLE", SystemMode.IDLE, (), "pass"),
    ],
)
def test_mode_is_uses_terminal_accepted_activation(
    expected: str,
    active: SystemMode | None,
    history: tuple[SystemMode, ...],
    status: str,
) -> None:
    """Published history cannot substitute for a current accepted activation."""
    capture = TelemetryCapture(
        inference_count=0,
        gimbal_moved=False,
        mode_activations=history,
        active_mode=active,
        acks=(),
        downlink_packets=(),
    )
    result = _score_frame_portable(
        Assertion(id="mode", kind="mode_is", value=expected, tag="frame-portable"), capture
    )
    assert result.status == status
    if active is None:
        assert "unknown" in result.detail
