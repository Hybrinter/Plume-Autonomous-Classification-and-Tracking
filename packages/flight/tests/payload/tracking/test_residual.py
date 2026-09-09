"""Tests for the two-state residual Kalman filter."""

import math

import numpy as np
from flight.libs.config import ControllerConfig
from flight.payload.tracking.residual import (
    EncoderSample,
    NominalRateSample,
    ObservationDisposition,
    PredictorReferenceChange,
    ResidualFilter,
    ResidualHistory,
    ResidualSnapshot,
    VisionObservation,
    estimate_at,
    predict,
    propagate_displacement,
    push_snapshot,
    rewind_update,
    submit_event,
    update,
)


def _filter(cfg: ControllerConfig | None = None) -> tuple[ControllerConfig, ResidualFilter]:
    """Build a residual filter from controller defaults."""
    resolved = cfg if cfg is not None else ControllerConfig()
    return resolved, ResidualFilter.from_config(resolved.residual, resolved.outer.dt_s)


def test_cold_state_is_zero() -> None:
    """The filter starts at e=0, omega_res=0, has_measurement=False."""
    _cfg, filt = _filter()
    state = filt.initial_state()
    assert state.has_measurement is False
    assert np.allclose(state.x, np.zeros(2))


def test_first_update_snaps_e() -> None:
    """The first vision update may snap e to z_v."""
    _cfg, filt = _filter()
    state = filt.initial_state()
    z_v = math.radians(-0.4)
    updated = update(filt, state, z_v)
    assert updated.has_measurement is True
    assert abs(float(updated.x[0]) - z_v) < 1e-15


def test_predict_ramps_error_from_rate_mismatch() -> None:
    """A residual rate mismatch integrates into e during predict."""
    _cfg, filt = _filter()
    state = filt.initial_state()
    dt = 0.02
    predicted = predict(filt, state, dt, omega_t_nom=0.1, y_m=0.0)
    assert abs(float(predicted.x[0]) - dt * 0.1) < 1e-12
    assert float(predicted.x[1]) == 0.0


def test_rewind_update_replays_to_now() -> None:
    """A lagged z_v restores a snapshot, updates, and replays to now."""
    cfg, filt = _filter()
    dt = cfg.outer.dt_s
    state = update(filt, filt.initial_state(), 0.0)
    snaps: tuple[ResidualSnapshot, ...] = ()
    now = 0.0
    omega = 0.1
    for _ in range(4):
        now += dt
        state = predict(filt, state, dt, omega, 0.0)
        snaps = push_snapshot(
            snaps,
            ResidualSnapshot(t_s=now, state=state, dt_s=dt, omega_t_nom=omega, y_m=0.0),
            cfg.residual.rewind_snapshots,
        )
    z_v = 0.01
    t_s = dt * 2
    rewound = rewind_update(filt, snaps, state, now, t_s, z_v, cfg.residual.rewind_horizon_s)
    naive = update(filt, state, z_v)
    assert rewound.has_measurement is True
    assert abs(float(rewound.x[0]) - float(naive.x[0])) > 1e-12

    snap = next(s for s in snaps if abs(s.t_s - t_s) < 1e-12)
    posterior = update(filt, snap.state, z_v)
    replay_from = t_s
    for later in snaps:
        if later.t_s <= t_s + 1e-12:
            continue
        step = later.t_s - replay_from
        posterior = predict(filt, posterior, step, later.omega_t_nom, later.y_m)
        replay_from = later.t_s
    tail = now - replay_from
    if tail > 1e-12:
        last = snaps[-1]
        posterior = predict(filt, posterior, tail, last.omega_t_nom, last.y_m)
    assert np.allclose(rewound.x, posterior.x)
    assert not np.allclose(rewound.x, state.x)


def test_rewind_before_oldest_snap_uses_inverse() -> None:
    """A shutter time before the oldest snapshot rolls back with F^{-1}."""
    cfg, filt = _filter()
    dt = cfg.outer.dt_s
    state = update(filt, filt.initial_state(), 0.0)
    snaps: tuple[ResidualSnapshot, ...] = ()
    now = 0.0
    omega = 0.05
    y_m = 0.01
    for _ in range(3):
        now += dt
        state = predict(filt, state, dt, omega, y_m)
        snaps = push_snapshot(
            snaps,
            ResidualSnapshot(t_s=now, state=state, dt_s=dt, omega_t_nom=omega, y_m=y_m),
            cfg.residual.rewind_snapshots,
        )
    t_s = snaps[0].t_s - 0.5 * dt
    z_v = 0.03
    rewound = rewind_update(filt, snaps, state, now, t_s, z_v, cfg.residual.rewind_horizon_s)
    pred = predict(filt, snaps[0].state, t_s - snaps[0].t_s, omega, y_m, add_q=False)
    posterior = update(filt, pred, z_v)
    replay_from = t_s
    for later in snaps:
        step = later.t_s - replay_from
        if step > 1e-12:
            posterior = predict(filt, posterior, step, later.omega_t_nom, later.y_m)
        replay_from = later.t_s
    tail = now - replay_from
    if tail > 1e-12:
        posterior = predict(filt, posterior, tail, snaps[-1].omega_t_nom, snaps[-1].y_m)
    assert np.allclose(rewound.x, posterior.x)
    noop = update(filt, snaps[0].state, z_v)
    assert abs(float(rewound.x[0]) - float(noop.x[0])) > 1e-8


def _history(
    filt: ResidualFilter,
    *,
    horizon_s: float = 10.0,
    anchor_angle_rad: float | None = 0.0,
    anchor_variance_rad2: float = 0.0,
    reversal_threshold_rad: float = 0.0,
    reversal_variance_rad2: float = 0.0,
    interpolation_span_max_s: float = math.inf,
) -> ResidualHistory:
    """Build a deterministic history with an optional encoder anchor."""
    return ResidualHistory.initial(
        filt.initial_state(),
        horizon_s=horizon_s,
        encoder_angle_rad=anchor_angle_rad,
        encoder_endpoint_variance_rad2=anchor_variance_rad2,
        reversal_threshold_rad=reversal_threshold_rad,
        reversal_variance_rad2=reversal_variance_rad2,
        interpolation_span_max_s=interpolation_span_max_s,
    )


def _submit(history: ResidualHistory, event: object) -> ResidualHistory:
    """Submit a known residual event and assert it was accepted."""
    updated, disposition = submit_event(history, event)  # type: ignore[arg-type]
    assert disposition is ObservationDisposition.ACCEPTED
    return updated


def test_displacement_propagation_is_independent_of_inner_rate_fit() -> None:
    """Encoder displacement, rather than a caller-derived y_m, drives elevation."""
    _cfg, filt = _filter()
    first = propagate_displacement(filt, filt.initial_state(), 0.5, 0.2, 0.1)
    second = propagate_displacement(filt, filt.initial_state(), 0.5, 0.2, 0.3)
    assert np.isclose(first.x[0] - second.x[0], 0.2)
    assert np.isclose(first.x[0], 0.1)


def test_continuous_process_covariance_has_partition_identity() -> None:
    """White-acceleration Q composes exactly when a span is partitioned."""
    _cfg, filt = _filter()
    whole = propagate_displacement(filt, filt.initial_state(), 1.0, 0.0, 0.0)
    half = propagate_displacement(filt, filt.initial_state(), 0.5, 0.0, 0.0)
    half = propagate_displacement(filt, half, 0.5, 0.0, 0.0)
    assert np.allclose(whole.x, half.x)
    assert np.allclose(whole.P, half.P)


def test_repeated_reporting_from_anchor_does_not_inflate_anchor_variance() -> None:
    """Replaying the same anchor span is deterministic and non-accumulating."""
    _cfg, filt = _filter()
    base = _history(filt, anchor_variance_rad2=0.04)
    one = _submit(base, EncoderSample("e1", 1.0, 1.0, 0.09))
    split = _submit(one, EncoderSample("e05", 0.5, 0.5, 0.09))
    first = estimate_at(one, filt, 1.0)
    repeated = estimate_at(one, filt, 1.0)
    partitioned = estimate_at(split, filt, 1.0)
    assert np.allclose(first.state.P, repeated.state.P)
    assert np.allclose(first.state.P, partitioned.state.P)
    assert np.isclose(float(first.state.P[0, 0]), filt.p0_11 + filt.p0_22 + 0.13 + filt.q_a / 3.0)


def test_one_reversal_adds_one_configured_covariance_charge() -> None:
    """A genuine direction reversal is charged once, not once per tick."""
    _cfg, filt = _filter()
    straight = _history(filt)
    straight = _submit(straight, EncoderSample("s1", 1.0, 1.0, 0.0))
    turning = _history(filt, reversal_threshold_rad=1e-6, reversal_variance_rad2=0.25)
    turning = _submit(turning, EncoderSample("s1", 0.5, 1.0, 0.0))
    turning = _submit(turning, EncoderSample("s2", 1.0, 0.5, 0.0))
    straight_state = estimate_at(straight, filt, 1.0).state
    turning_state = estimate_at(turning, filt, 1.0).state
    assert np.isclose(turning_state.P[0, 0] - straight_state.P[0, 0], 0.25)


def test_nominal_event_does_not_hide_encoder_reversal() -> None:
    """A causal nominal-rate event cannot partition away reversal covariance."""
    _cfg, filt = _filter()
    history = _history(filt, reversal_threshold_rad=1e-6, reversal_variance_rad2=0.25)
    history = _submit(history, EncoderSample("up", 0.4, 1.0))
    history = _submit(history, NominalRateSample("nominal", 0.5, 0.0))
    history = _submit(history, EncoderSample("down", 1.0, 0.5))
    result = estimate_at(history, filt, 1.0)
    baseline = propagate_displacement(filt, filt.initial_state(), 1.0, 0.0, 0.5)
    assert np.isclose(result.state.P[0, 0] - baseline.P[0, 0], 0.25)


def test_wide_encoder_bracket_rejects_vision() -> None:
    """Shutter interpolation cannot exceed the configured span."""
    _cfg, filt = _filter()
    history = _history(filt, interpolation_span_max_s=0.2)
    history = _submit(history, EncoderSample("left", 0.0, 0.0))
    history = _submit(history, EncoderSample("right", 1.0, 1.0))
    history = _submit(history, VisionObservation("wide", 0.5, 0.1, 1e-6))
    result = estimate_at(history, filt, 1.0)
    assert result.dispositions[0].disposition is ObservationDisposition.NO_ENCODER_BRACKET


def test_nominal_rate_is_causal_zoh_and_reference_rebase_preserves_total_rate() -> None:
    """Nominal samples are left-held; explicit replacement rebases residual rate."""
    _cfg, filt = _filter()
    history = _history(filt)
    history = _submit(history, EncoderSample("e2", 2.0, 0.0))
    history = _submit(history, NominalRateSample("n1", 0.5, 1.0))
    history = _submit(history, NominalRateSample("n2", 1.5, 2.0))
    zoh = estimate_at(history, filt, 2.0).state
    assert np.isclose(zoh.x[0], 2.0)

    replacement = _history(filt)
    replacement = _submit(replacement, EncoderSample("e2", 2.0, 0.0))
    replacement = _submit(replacement, NominalRateSample("n0", 0.0, 1.0))
    replacement = _submit(
        replacement,
        PredictorReferenceChange("ref", 1.0, old_rate_rad_s=1.0, new_rate_rad_s=3.0),
    )
    rebased = estimate_at(replacement, filt, 2.0).state
    assert np.isclose(rebased.x[0], 2.0)
    assert np.isclose(rebased.x[1], -2.0)


def test_bracket_dispositions_duplicate_future_and_expired_events() -> None:
    """Vision requires a valid encoder bracket and IDs are deduplicated."""
    _cfg, filt = _filter()
    history = _history(filt, horizon_s=0.5)
    vision = VisionObservation("frame-1", 0.25, 0.1, 1e-6)
    history = _submit(history, vision)
    no_bracket = estimate_at(history, filt, 0.25)
    assert no_bracket.dispositions[0].disposition is ObservationDisposition.NO_ENCODER_BRACKET

    history, duplicate = submit_event(history, vision)
    assert duplicate is ObservationDisposition.DUPLICATE
    history = _submit(history, EncoderSample("e0", 0.0, 0.0))
    history = _submit(history, EncoderSample("e1", 0.5, 0.0))
    accepted = estimate_at(history, filt, 0.5)
    assert accepted.dispositions[0].disposition is ObservationDisposition.ACCEPTED

    future = _submit(history, VisionObservation("future", 1.0, 0.0, 1e-6))
    future_result = estimate_at(future, filt, 0.5)
    assert any(d.disposition is ObservationDisposition.FUTURE for d in future_result.dispositions)

    old = VisionObservation("old", -1.0, 0.0, 1e-6)
    _unchanged, expired = submit_event(history, old)
    assert expired is ObservationDisposition.EXPIRED


def test_delayed_replay_and_horizon_trim_are_deterministic() -> None:
    """A late vision correction survives trimming through its posterior anchor."""
    _cfg, filt = _filter()
    history = _history(filt, horizon_s=1.0)
    for event in (
        EncoderSample("e1", 1.0, 0.0),
        EncoderSample("e2", 2.0, 0.0),
        EncoderSample("e3", 3.0, 0.0),
    ):
        history = _submit(history, event)
    before = estimate_at(history, filt, 3.0)
    history = _submit(history, VisionObservation("late", 1.0, 0.2, 1e-6))
    after = estimate_at(history, filt, 3.0)
    assert not np.allclose(before.state.x, after.state.x)
    assert after.history.checkpoint.t_s == 2.0
    repeated = estimate_at(after.history, filt, 3.0)
    assert np.allclose(after.state.x, repeated.state.x)
    assert np.allclose(after.state.P, repeated.state.P)

    _unchanged, disposition = submit_event(
        after.history,
        VisionObservation("before-checkpoint", 1.5, 0.0, 1e-6),
        now_s=3.0,
    )
    assert disposition is ObservationDisposition.EXPIRED


def test_boundary_events_are_absorbed_once_by_trim_checkpoint() -> None:
    """An event exactly at a trim boundary is not replayed from its posterior."""
    _cfg, filt = _filter()
    history = _history(filt, horizon_s=1.0)
    history = _submit(history, EncoderSample("e1", 1.0, 0.0))
    history = _submit(history, EncoderSample("e2", 2.0, 0.0))
    history = _submit(history, VisionObservation("at-boundary", 1.0, 0.2, 1e-6))
    first = estimate_at(history, filt, 2.0)
    repeated = estimate_at(first.history, filt, 2.0)
    assert np.allclose(first.state.x, repeated.state.x)
    assert np.allclose(first.state.P, repeated.state.P)
