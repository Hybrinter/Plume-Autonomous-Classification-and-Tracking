"""Chronology, covariance, predictor-transition, and optical oracle tests."""

from __future__ import annotations

import numpy as np
from analysis.studies.tracking_estimator.chronology import (
    EventHistory,
    residual_chronological_oracle,
)
from analysis.studies.tracking_estimator.model import (
    CorrectedResidualEstimator,
    PredictorTransitionEvent,
    PropagationEvent,
    VisionEvent,
    continuous_kinematic_noise,
)
from analysis.studies.tracking_estimator.optical import PlumeRegion, integrate_exposure
from analysis.studies.tracking_estimator.scenarios import (
    run_estimator_comparison,
    run_optical_exposure_sweep,
)


def test_continuous_noise_is_partition_consistent() -> None:
    """Q over 0.17 + 0.23 s equals Q over the combined interval."""
    f_a, q_a = continuous_kinematic_noise(0.17, 3.0e-7)
    f_b, q_b = continuous_kinematic_noise(0.23, 3.0e-7)
    _f_full, q_full = continuous_kinematic_noise(0.40, 3.0e-7)
    assert np.allclose(f_b @ q_a @ f_b.T + q_b, q_full)
    assert np.all(np.linalg.eigvalsh(q_full) >= -1.0e-18)
    assert f_a.shape == (2, 2)


def test_delayed_residual_history_matches_independent_chronological_oracle() -> None:
    """Late vision preserves every accepted prior correction after replay."""
    events = (
        PropagationEvent("p0", 0.0, None, 0.0),
        PropagationEvent("p1", 0.20, 0.010, 0.003),
        VisionEvent("v1", 0.20, 0.004),
        PropagationEvent("p2", 0.43, 0.010, 0.003),
        VisionEvent("v2", 0.43, 0.006),
        PropagationEvent("p3", 0.71, 0.010, 0.004),
        PredictorTransitionEvent("ref", 0.71, 0.010, 0.012),
        VisionEvent("v3", 0.71, 0.007),
    )
    history = EventHistory(CorrectedResidualEstimator, horizon_s=2.0)
    arrivals = (
        events[0],
        events[1],
        events[3],
        events[4],
        events[5],
        events[6],
        events[7],
        events[2],
    )
    for item in arrivals:
        history.submit(item, now_s=0.72)
    actual = history.replay().estimator.estimate
    oracle = residual_chronological_oracle(events, 3.0e-7, 2.5e-7)
    assert np.allclose(actual.covariance, oracle.p)
    assert actual.relative_angle_rad == np.float64(oracle.x[0])
    assert actual.target_rate_rad_s == np.float64(oracle.predictor_rate_rad_s + oracle.x[1])


def test_predictor_reference_change_keeps_total_rate_continuous() -> None:
    """Changing nominal rate changes residual representation but not total rate."""
    estimator = CorrectedResidualEstimator()
    estimator.apply(PropagationEvent("p0", 0.0, 0.010, 0.0))
    estimator.apply(PropagationEvent("p1", 1.0, 0.010, 0.0))
    before = estimator.estimate.target_rate_rad_s
    estimator.apply(PredictorTransitionEvent("switch", 1.0, 0.010, 0.016))
    assert estimator.estimate.target_rate_rad_s == before


def test_duplicate_and_history_expiry_are_explicit() -> None:
    """Duplicates and stale history are reported rather than silently re-applied."""
    history = EventHistory(CorrectedResidualEstimator, horizon_s=0.1)
    event = PropagationEvent("p0", 0.0, None, 0.0)
    history.submit(event, now_s=0.0)
    history.submit(event, now_s=0.0)
    history.submit(VisionEvent("old", 0.0, 0.1), now_s=1.0)
    result = history.replay()
    assert result.duplicate_event_ids == ("p0",)
    assert result.expired_event_ids == ("old",)


def test_image_formation_reports_local_blur_not_just_centroid_motion() -> None:
    """A shearing region can violate blur while aggregate centroid remains centered."""
    image = integrate_exposure(
        (
            PlumeRegion("slow", 32.0, 32.0, 0.0, 0.0, 2.0, 20.0),
            PlumeRegion("fast", 32.0, 32.0, 0.0, 3_000.0, 2.0, 20.0),
        ),
        (64, 64),
        1.0e-3,
        0.0,
        1.0e-4,
        smear_budget_px=1.0,
    )
    assert image.centroid_y_px is not None
    assert image.local_blur[0].sharp_detected_area_px == 20.0
    assert image.local_blur[1].sharp_detected_area_px == 0.0


def test_selection_checkpoint_runs_both_candidates_and_exposure_anchors() -> None:
    """Comparison stays unselected and includes 13 us and 1 ms image cases."""
    comparison = run_estimator_comparison()
    sweep = run_optical_exposure_sweep()
    assert comparison.selection_status.startswith("UNSELECTED")
    assert comparison.residual.accepted_events > 0
    assert comparison.joint.accepted_events > 0
    assert {item.exposure_us for item in sweep} == {13.0, 1_000.0}
    assert {item.estimator_name for item in sweep} == {
        "CorrectedResidualEstimator",
        "JointAngularEstimator",
    }
    assert all(item.area_weighted_p90_blur_px <= item.worst_local_blur_px for item in sweep)
