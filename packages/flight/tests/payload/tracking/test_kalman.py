"""Tests for the Kalman pointing estimator."""

import numpy as np
from flight.libs.config import ControllerConfig
from flight.libs.types import Ok
from flight.payload.tracking import KalmanFilter, predict, update


def test_predict_keeps_state_shape() -> None:
    """predict returns a KalmanState whose estimate is the 4-vector [pan, tilt, dpan, dtilt]."""
    kf = KalmanFilter.from_config(ControllerConfig())
    state = KalmanFilter.initial_state(pan_deg=0.0, tilt_deg=0.0)
    predicted = predict(kf, state)
    estimate = np.asarray(predicted.x)
    assert estimate.shape == (4,)


def test_update_incorporates_observation() -> None:
    """update returns Ok(KalmanState) for a finite 2D observation."""
    kf = KalmanFilter.from_config(ControllerConfig())
    state = KalmanFilter.initial_state(pan_deg=1.0, tilt_deg=2.0)
    result = update(kf, state, np.array([1.5, 2.5], dtype=np.float64))
    assert isinstance(result, Ok)
    assert np.asarray(result.value.x).shape == (4,)
