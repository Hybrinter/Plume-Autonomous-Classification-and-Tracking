"""2D linear Kalman filter for gimbal pointing state estimation.

State vector: [pan_deg, tilt_deg, pan_rate_deg_s, tilt_rate_deg_s]
Observation:  [pan_deg, tilt_deg]

Satisfies: REQ-GIMB-HIGH-002 (stable and bounded pointing behavior).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pact.types.config import ControllerConfig
from pact.types.enums import FaultCode, Ok, Err


@dataclass(frozen=True)
class KalmanState:
    """Immutable Kalman filter state."""

    x: object  # np.ndarray[float64, (4,)] -- state estimate
    P: object  # np.ndarray[float64, (4,4)] -- error covariance


@dataclass(frozen=True)
class KalmanFilter:
    """Constant-velocity Kalman filter matrices for 2-axis gimbal tracking.

    F: state transition matrix (4x4)
    H: observation matrix (2x4)
    Q: process noise covariance (4x4)
    R: measurement noise covariance (2x2)
    """

    F: object  # np.ndarray[float64, (4,4)]
    H: object  # np.ndarray[float64, (2,4)]
    Q: object  # np.ndarray[float64, (4,4)]
    R: object  # np.ndarray[float64, (2,2)]

    @staticmethod
    def from_config(cfg: ControllerConfig) -> KalmanFilter:
        """Build filter matrices from ControllerConfig."""
        dt = cfg.kalman_dt_s
        q = cfg.kalman_process_noise
        r = cfg.kalman_measurement_noise
        # Constant-velocity model: position updated by velocity * dt
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        Q = np.eye(4, dtype=np.float64) * q
        R = np.eye(2, dtype=np.float64) * r
        return KalmanFilter(F=F, H=H, Q=Q, R=R)

    @staticmethod
    def initial_state(
        pan_deg: float, tilt_deg: float,
    ) -> KalmanState:
        """Create an initial Kalman state at a given position."""
        x = np.array(
            [pan_deg, tilt_deg, 0.0, 0.0], dtype=np.float64,
        )
        P = np.eye(4, dtype=np.float64) * 1.0
        return KalmanState(x=x, P=P)


def predict(kf: KalmanFilter, state: KalmanState) -> KalmanState:
    """Kalman predict step: propagate state forward by dt."""
    x_pred = kf.F @ state.x
    P_pred = kf.F @ state.P @ kf.F.T + kf.Q
    return KalmanState(x=x_pred, P=P_pred)


def update(
    kf: KalmanFilter,
    state: KalmanState,
    observation: object,  # np.ndarray[float64, (2,)]
) -> "Ok[KalmanState] | Err[FaultCode]":
    """Kalman update step: incorporate observation [pan_deg, tilt_deg].

    Returns Err(GIMBAL_RUNAWAY) if innovation covariance is singular.
    """
    # Innovation covariance (2x2)
    S = kf.H @ state.P @ kf.H.T + kf.R
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return Err(FaultCode.GIMBAL_RUNAWAY)
    # Kalman gain (4x2)
    K = state.P @ kf.H.T @ S_inv
    innovation = observation - kf.H @ state.x
    x_upd = state.x + K @ innovation
    P_upd = (np.eye(4, dtype=np.float64) - K @ kf.H) @ state.P
    return Ok(KalmanState(x=x_upd, P=P_upd))
