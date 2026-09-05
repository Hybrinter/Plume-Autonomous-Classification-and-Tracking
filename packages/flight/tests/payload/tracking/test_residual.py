"""Tests for the two-state residual Kalman filter."""

import math

import numpy as np
from flight.libs.config import ControllerConfig
from flight.payload.tracking.residual import (
    ResidualFilter,
    ResidualSnapshot,
    predict,
    push_snapshot,
    rewind_update,
    update,
)


def test_cold_state_is_zero() -> None:
    """The filter starts at e=0, omega_res=0, has_measurement=False."""
    filt = ResidualFilter.from_config(ControllerConfig())
    state = filt.initial_state()
    assert state.has_measurement is False
    assert np.allclose(state.x, np.zeros(2))


def test_first_update_snaps_e() -> None:
    """The first vision update may snap e to z_v."""
    filt = ResidualFilter.from_config(ControllerConfig())
    state = filt.initial_state()
    z_v = math.radians(-0.4)
    updated = update(filt, state, z_v)
    assert updated.has_measurement is True
    assert abs(float(updated.x[0]) - z_v) < 1e-15


def test_predict_ramps_error_from_rate_mismatch() -> None:
    """A residual rate mismatch integrates into e during predict."""
    filt = ResidualFilter.from_config(ControllerConfig())
    state = filt.initial_state()
    dt = 0.02
    predicted = predict(filt, state, dt, omega_t_nom=0.1, y_m=0.0)
    assert abs(float(predicted.x[0]) - dt * 0.1) < 1e-12
    assert float(predicted.x[1]) == 0.0


def test_rewind_update_replays_to_now() -> None:
    """A lagged z_v restores a snapshot, updates, and replays to now."""
    cfg = ControllerConfig()
    filt = ResidualFilter.from_config(cfg)
    dt = cfg.dt_outer_s
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
            cfg.rewind_snapshots,
        )
    z_v = 0.01
    t_s = dt * 2
    rewound = rewind_update(filt, snaps, state, now, t_s, z_v, cfg.rewind_horizon_s)
    naive = update(filt, state, z_v)
    assert rewound.has_measurement is True
    # Lagged update must not equal treating z_v as a measurement at `now`.
    assert abs(float(rewound.x[0]) - float(naive.x[0])) > 1e-12
