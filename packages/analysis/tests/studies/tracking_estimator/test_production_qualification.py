"""Qualification checks comparing flight replay with an independent oracle.

These tests intentionally live in ``analysis``.  They are evidence checks, not
runtime estimator selection logic.  The oracle has its own event replay,
interpolation, covariance, and trim implementation in ``oracle.py``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from analysis.studies.tracking_estimator.model import (
    EncoderEvent,
    PredictorTransitionEvent,
    PropagationEvent,
    TimelineEvent,
    VisionEvent,
)
from analysis.studies.tracking_estimator.oracle import (
    AnchoredOracleCheckpoint,
    AnchoredOracleJournal,
)
from flight.payload.tracking.residual import (
    EncoderSample,
    NominalRateSample,
    PredictorReferenceChange,
    ResidualEvent,
    ResidualFilter,
    ResidualHistory,
    VisionObservation,
    estimate_at,
    propagate_displacement,
    submit_event,
    update,
)

_ACCEL_PSD = 3.0e-7
_VISION_VARIANCE = 2.5e-7
_ENCODER_VARIANCE = 1.0e-7
_REVERSAL_VARIANCE = 8.0e-8
_HORIZON_S = 0.55


def _timeline(seed: int) -> tuple[TimelineEvent, ...]:
    """Build nonuniform increments, reversals, nominal samples, and visions."""
    rng = np.random.default_rng(seed)
    times = np.concatenate(([0.0], np.cumsum(rng.uniform(0.035, 0.085, size=44))))
    angle = 0.0
    nominal = 0.004
    events: list[TimelineEvent] = []
    for index, t_s in enumerate(times):
        if index:
            delta = float(rng.normal(0.0, 0.0015))
            if index % 9 == 0:
                delta = -abs(delta) if index % 18 else abs(delta)
            angle += delta
        sampled_nominal: float | None = nominal if index == 0 or index % 7 == 0 else None
        if sampled_nominal is not None:
            events.append(
                PropagationEvent(
                    f"p-{seed}-{index}",
                    float(t_s),
                    sampled_nominal,
                    0.0,
                )
            )
        events.append(EncoderEvent(f"e-{seed}-{index}", float(t_s), angle))
        if index in (15, 30):
            next_nominal = nominal + (0.002 if index == 15 else -0.003)
            events.append(
                PredictorTransitionEvent(
                    f"r-{seed}-{index}",
                    float(t_s),
                    nominal,
                    next_nominal,
                )
            )
            nominal = next_nominal
        if index >= 4 and index % 4 == 0:
            truth_error = 0.002 * np.sin(0.65 * t_s) + 0.0005 * np.cos(0.21 * t_s)
            noisy_error = float(truth_error + rng.normal(0.0, np.sqrt(_VISION_VARIANCE)))
            events.append(VisionEvent(f"v-{seed}-{index}", float(t_s), noisy_error))
    return tuple(events)


def _production_event(event: TimelineEvent) -> ResidualEvent | None:
    """Translate analysis records into immutable flight events only."""
    if isinstance(event, EncoderEvent):
        return EncoderSample(event.event_id, event.t_s, event.angle_rad, _ENCODER_VARIANCE)
    if isinstance(event, PropagationEvent) and event.nominal_target_rate_rad_s is not None:
        return NominalRateSample(event.event_id, event.t_s, event.nominal_target_rate_rad_s)
    if isinstance(event, PredictorTransitionEvent):
        return PredictorReferenceChange(
            event.event_id,
            event.t_s,
            0.0 if event.previous_rate_rad_s is None else event.previous_rate_rad_s,
            0.0 if event.next_rate_rad_s is None else event.next_rate_rad_s,
        )
    if isinstance(event, VisionEvent):
        return VisionObservation(
            event.event_id, event.t_s, event.relative_angle_rad, _VISION_VARIANCE
        )
    return None


def _production_history(filt: ResidualFilter) -> ResidualHistory:
    """Start after a valid initial acquisition to exercise Joseph updates."""
    initial = replace(filt.initial_state(), has_measurement=True)
    return ResidualHistory.initial(
        initial,
        horizon_s=_HORIZON_S,
        encoder_angle_rad=0.0,
        encoder_endpoint_variance_rad2=_ENCODER_VARIANCE,
        reversal_threshold_rad=1.0e-8,
        reversal_variance_rad2=_REVERSAL_VARIANCE,
    )


def _qualification_run(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one delayed-arrival seed through production and the oracle."""
    events = _timeline(seed)
    filt = ResidualFilter(
        q_a=_ACCEL_PSD,
        r_v=_VISION_VARIANCE,
        p0_11=1.0e-3,
        p0_22=1.0e-4,
        dt_outer_s=0.02,
        rewind_horizon_s=_HORIZON_S,
    )
    production = _production_history(filt)
    oracle = AnchoredOracleJournal(
        _HORIZON_S,
        AnchoredOracleCheckpoint(
            0.0,
            np.zeros(2, dtype=np.float64),
            np.diag(np.array([1.0e-3, 1.0e-4], dtype=np.float64)),
            0.0,
            _ENCODER_VARIANCE,
            0.0,
        ),
        reversal_variance_rad2=_REVERSAL_VARIANCE,
    )
    rng = np.random.default_rng(seed + 100_000)
    arrivals: list[tuple[float, int, TimelineEvent]] = []
    for index, event in enumerate(events):
        delay = float(rng.uniform(0.0, 0.18)) if isinstance(event, VisionEvent) else 0.0
        arrivals.append((event.t_s + delay, index, event))
    arrivals.sort(key=lambda item: (item[0], item[1], item[2].event_id))
    latest_endpoint = 0.0
    for arrival_s, _index, event in arrivals:
        converted = _production_event(event)
        if converted is not None:
            production, _ = submit_event(production, converted)
        if isinstance(event, EncoderEvent):
            latest_endpoint = max(latest_endpoint, event.t_s)
        oracle.submit(event, arrival_s)
        production_result = estimate_at(production, filt, latest_endpoint)
        production = production_result.history
        oracle.estimate(latest_endpoint)

    final_t = max(event.t_s for event in events if isinstance(event, EncoderEvent))
    production_result = estimate_at(production, filt, final_t)
    oracle_result = oracle.estimate(final_t)
    return (
        production_result.state.x,
        oracle_result.state.x,
        production_result.state.P,
        oracle_result.state.covariance,
    )


def test_32_seed_delayed_replay_matches_independent_oracle() -> None:
    """Every deterministic qualification seed matches means/covariance after trims."""
    for seed in range(32):
        actual_x, expected_x, actual_p, expected_p = _qualification_run(seed)
        assert np.allclose(actual_x, expected_x, atol=2.0e-11), seed
        assert np.allclose(actual_p, expected_p, atol=2.0e-11), seed
        assert np.all(np.linalg.eigvalsh(actual_p) >= -1.0e-12), seed


def test_monte_carlo_innovations_are_covered_and_psd() -> None:
    """Thirty-two paired streams keep normalized innovations and PSD covariances sane."""
    normalized: list[float] = []
    for seed in range(32):
        rng = np.random.default_rng(200_000 + seed)
        filt = ResidualFilter(
            q_a=_ACCEL_PSD,
            r_v=_VISION_VARIANCE,
            p0_11=1.0e-3,
            p0_22=1.0e-4,
            dt_outer_s=0.02,
        )
        state = filt.initial_state()
        truth = np.zeros(2, dtype=np.float64)
        for _step in range(48):
            dt_s = float(rng.uniform(0.01, 0.05))
            acceleration = float(rng.normal(0.0, np.sqrt(_ACCEL_PSD / dt_s)))
            truth[0] += truth[1] * dt_s + 0.5 * acceleration * dt_s**2
            truth[1] += acceleration * dt_s
            predicted = propagate_displacement(filt, state, dt_s, 0.0, 0.0)
            innovation = float(
                truth[0] + rng.normal(0.0, np.sqrt(_VISION_VARIANCE)) - predicted.x[0]
            )
            innovation_covariance = float(predicted.P[0, 0] + _VISION_VARIANCE)
            normalized.append(innovation * innovation / innovation_covariance)
            state = update(
                filt,
                predicted,
                truth[0] + rng.normal(0.0, np.sqrt(_VISION_VARIANCE)),
                _VISION_VARIANCE,
            )
            assert np.all(np.linalg.eigvalsh(state.P) >= -1.0e-12)
    nis_mean = float(np.mean(normalized[32:]))
    assert 0.75 < nis_mean < 1.25
