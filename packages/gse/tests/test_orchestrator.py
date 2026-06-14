"""run_scenario scores frame-portable assertions and skips realtime-only ones."""

from gse.orchestrator import run_scenario
from gse.scenario import Assertion, Scenario, SceneSpec


def _scored_scenario() -> Scenario:
    """All-sim scenario asserting a moved gimbal, an inference floor, and a skipped timing one."""
    return Scenario(
        name="orchestrator-smoke",
        profile="profiles/sil.toml",
        scene=SceneSpec(num_frames=6, seed=0),
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
        steps=6,
        dt=1.0,
    )


def test_run_scenario_scores_and_skips() -> None:
    """Frame-portable assertions pass; the realtime-only assertion is skipped with a reason."""
    report = run_scenario(_scored_scenario(), "profiles/sil.toml")

    assert report.scenario == "orchestrator-smoke"
    assert report.passed == 2
    assert report.failed == 0
    assert report.skipped == 1

    by_id = {r.id: r for r in report.results}
    assert by_id["GIMBAL-MOVED"].status == "pass"
    assert by_id["INF-FLOOR"].status == "pass"
    assert by_id["ACK-TIMING"].status == "skip"
    assert "realtime-only" in by_id["ACK-TIMING"].detail
