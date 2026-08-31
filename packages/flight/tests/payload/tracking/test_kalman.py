"""Tests for the per-axis 4-state Kalman pointing estimator."""

import numpy as np
from flight.libs.config import ControllerConfig
from flight.libs.types import Ok
from flight.payload.tracking import KalmanFilter, predict, update_enc, update_vis


def test_predict_keeps_axis_shape() -> None:
    """predict returns a 4-vector [e, theta_g, omega_t, omega_g]."""
    kf = KalmanFilter.from_config(ControllerConfig())
    state = KalmanFilter.initial_axis(e_deg=0.0, theta_g_deg=0.0)
    predicted = predict(kf, state, u=0.0, dt=0.05)
    assert np.asarray(predicted.x).shape == (4,)


def test_update_vis_measures_error() -> None:
    """A vision update pulls e toward the measured boresight error."""
    kf = KalmanFilter.from_config(ControllerConfig())
    state = KalmanFilter.initial_axis(e_deg=0.0, theta_g_deg=0.0)
    result = update_vis(kf, state, z_vis=2.0)
    assert isinstance(result, Ok)
    assert result.value.x[0] > 0.0


def test_update_enc_measures_gimbal_angle() -> None:
    """An encoder update pulls theta_g toward the measured shaft angle."""
    kf = KalmanFilter.from_config(ControllerConfig())
    state = KalmanFilter.initial_axis(e_deg=0.0, theta_g_deg=0.0)
    result = update_enc(kf, state, theta_enc_deg=5.0)
    assert isinstance(result, Ok)
    assert result.value.x[1] > 0.0


def test_predict_with_rate_command_moves_theta_g() -> None:
    """A positive rate command increases estimated gimbal angle and reduces e."""
    kf = KalmanFilter.from_config(ControllerConfig())
    state = KalmanFilter.initial_axis(e_deg=1.0, theta_g_deg=0.0)
    predicted = predict(kf, state, u=2.0, dt=0.05)
    assert predicted.x[1] > state.x[1]
    assert predicted.x[0] < state.x[0]
