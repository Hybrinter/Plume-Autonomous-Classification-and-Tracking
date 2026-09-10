"""Independent chronological acceptance tests for the residual estimator plan."""

from __future__ import annotations

import numpy as np
from analysis.studies.tracking_estimator.model import (
    EncoderEvent,
    PredictorTransitionEvent,
    PropagationEvent,
    VisionEvent,
)
from analysis.studies.tracking_estimator.oracle import CheckpointedOracle, replay_residual


def _timeline() -> tuple[PropagationEvent | EncoderEvent | VisionEvent, ...]:
    """Return a nonuniform timeline with delayed visual corrections."""
    return (
        PropagationEvent("p0", 0.0, 0.020, 0.000),
        EncoderEvent("e0", 0.0, 0.000),
        PropagationEvent("p1", 0.13, None, 0.004),
        EncoderEvent("e1", 0.13, 0.001),
        VisionEvent("v1", 0.13, 0.008),
        PropagationEvent("p2", 0.31, None, 0.007),
        EncoderEvent("e2", 0.31, 0.002),
        VisionEvent("v2", 0.22, 0.005),
        PropagationEvent("p3", 0.47, None, 0.002),
        EncoderEvent("e3", 0.47, 0.004),
    )


def test_out_of_order_vision_matches_chronological_oracle() -> None:
    """Arrival order does not alter the physical-time posterior."""
    events = _timeline()
    history = CheckpointedOracle(horizon_s=2.0)
    # The second vision is deliberately delivered after later propagation and
    # encoder events; sorting is by event time, causal priority, then ID.
    arrival_order = (0, 1, 2, 3, 4, 5, 6, 8, 7, 9)
    for index in arrival_order:
        event = events[index]
        assert history.submit(event, arrival_s=0.5)
    actual = history.estimate()
    expected = replay_residual(events)
    assert np.allclose(actual.x, expected.x, atol=1.0e-14)
    assert np.allclose(actual.covariance, expected.covariance, atol=1.0e-14)
    assert actual.nominal_rate_rad_s == expected.nominal_rate_rad_s


def test_repeated_trims_preserve_prior_vision_correction() -> None:
    """Trimming advances a checkpoint instead of cold-starting from the prior."""
    events = (
        PropagationEvent("p0", 0.0, 0.03, 0.0),
        VisionEvent("v0", 0.0, 0.12),
        PropagationEvent("p1", 0.10, None, 0.0),
        PropagationEvent("p2", 0.20, None, 0.0),
        PropagationEvent("p3", 0.30, None, 0.0),
        PropagationEvent("p4", 0.40, None, 0.0),
        PropagationEvent("p5", 0.50, None, 0.0),
        PropagationEvent("p6", 0.60, None, 0.0),
        PropagationEvent("p7", 0.70, None, 0.0),
        PropagationEvent("p8", 0.80, None, 0.0),
    )
    history = CheckpointedOracle(horizon_s=0.30)
    for event in events:
        assert history.submit(event, arrival_s=event.t_s)
    actual = history.estimate()
    expected = replay_residual(events)
    assert history.checkpoint.t_s >= 0.50 - 1.0e-12
    assert float(actual.x[0]) > 0.05
    assert np.allclose(actual.x, expected.x, atol=1.0e-14)
    assert np.allclose(actual.covariance, expected.covariance, atol=1.0e-14)


def test_nominal_rate_is_zero_order_hold_until_explicit_replacement() -> None:
    """Sampled nominal evolution holds its prior rate; replacement rebases residual."""
    events = (
        PropagationEvent("p0", 0.0, 0.10, 0.0),
        PropagationEvent("p1", 0.20, None, 0.0),
        PropagationEvent("p2", 0.45, None, 0.0),
        PredictorTransitionEvent("ref", 0.45, 0.10, 0.18),
        PropagationEvent("p3", 0.70, None, 0.0),
    )
    result = replay_residual(events)
    # Before the explicit replacement, the nominal rate is held at 0.10.  The
    # replacement changes residual coordinates only, keeping total rate 0.10.
    assert result.nominal_rate_rad_s == 0.18
    assert np.isclose(result.x[1], -0.08)
    assert np.isclose(result.nominal_rate_rad_s + result.x[1], 0.10)


def test_partitioned_process_noise_matches_one_interval() -> None:
    """The white-acceleration covariance obeys the time-partition identity."""
    whole = replay_residual(
        (
            PropagationEvent("p0", 0.0, 0.0, 0.0),
            PropagationEvent("p1", 0.40, None, 0.0),
        )
    )
    split = replay_residual(
        (
            PropagationEvent("p0", 0.0, 0.0, 0.0),
            PropagationEvent("p1", 0.17, None, 0.0),
            PropagationEvent("p2", 0.40, None, 0.0),
        )
    )
    assert np.allclose(split.x, whole.x, atol=1.0e-14)
    assert np.allclose(split.covariance, whole.covariance, atol=1.0e-14)


def test_duplicate_and_expired_events_are_dispositioned() -> None:
    """Stable IDs are deduplicated and events older than the horizon expire."""
    history = CheckpointedOracle(horizon_s=0.1)
    first = PropagationEvent("p0", 0.0, 0.0, 0.0)
    assert history.submit(first, arrival_s=0.0)
    assert not history.submit(first, arrival_s=0.0)
    assert not history.submit(VisionEvent("old", 0.0, 0.1), arrival_s=1.0)
    assert history.duplicate_ids == ["p0"]
    assert history.expired_ids == ["old"]
