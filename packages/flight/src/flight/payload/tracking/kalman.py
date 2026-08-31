"""Per-axis 4-state linear Kalman filter for gimbal pointing (pure).

State per axis: [e, theta_g, omega_t, omega_g] in degrees and deg/s.
e is boresight error (theta_target - theta_g). Vision measures e. Encoder
measures theta_g. Predict uses the applied rate command u (deg/s) and the
actual dt.

Satisfies: REQ-GIMB-HIGH-002.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from flight.libs.config import ControllerConfig
from flight.libs.types import Err, FaultCode, Ok


def continuous_plant(tau_cl_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (A_c, B_c) for x = [e, theta_g, omega_t, omega_g], u = rate command.

    Inner closed loop is modeled as omega_g_dot = (-omega_g + u) / tau_cl_s.
    """
    inv_tau = 1.0 / tau_cl_s
    a_c = np.array(
        [
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -inv_tau],
        ],
        dtype=np.float64,
    )
    b_c = np.array([[0.0], [0.0], [0.0], [inv_tau]], dtype=np.float64)
    return a_c, b_c


def discretize(a_c: np.ndarray, b_c: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact discrete (Phi, B_d) via the Van Loan exponential of the augmented plant."""
    n = a_c.shape[0]
    m = np.zeros((n + 1, n + 1), dtype=np.float64)
    m[:n, :n] = a_c * dt
    m[:n, n : n + 1] = b_c * dt
    expm = scipy.linalg.expm(m)
    phi = expm[:n, :n]
    b_d = expm[:n, n : n + 1]
    return phi, b_d


@dataclass(frozen=True, slots=True)
class AxisKalmanState:
    """Immutable 4-state Kalman snapshot for one gimbal axis."""

    x: np.ndarray  # np.ndarray[float64, (4,)] -- [e, theta_g, omega_t, omega_g]
    p: np.ndarray  # np.ndarray[float64, (4, 4)] -- covariance


@dataclass(frozen=True, slots=True)
class DualKalmanState:
    """Azimuth and elevation Kalman snapshots."""

    az: AxisKalmanState
    el: AxisKalmanState


@dataclass(frozen=True, slots=True)
class KalmanFilter:
    """Continuous plant plus measurement covariances shared by both axes."""

    a_c: np.ndarray  # np.ndarray[float64, (4, 4)]
    b_c: np.ndarray  # np.ndarray[float64, (4, 1)]
    h_vis: np.ndarray  # np.ndarray[float64, (1, 4)]
    h_enc: np.ndarray  # np.ndarray[float64, (1, 4)]
    q: np.ndarray  # np.ndarray[float64, (4, 4)] -- continuous process density
    r_vis: float
    r_enc: float

    @staticmethod
    def from_config(cfg: ControllerConfig) -> KalmanFilter:
        """Build the shared plant and noise matrices from ControllerConfig."""
        a_c, b_c = continuous_plant(cfg.tau_cl_s)
        q_scale = cfg.kalman_process_noise
        q = np.diag(np.array([q_scale, 1e-8, q_scale, q_scale * 0.1], dtype=np.float64))
        h_vis = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        h_enc = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        return KalmanFilter(
            a_c=a_c,
            b_c=b_c,
            h_vis=h_vis,
            h_enc=h_enc,
            q=q,
            r_vis=cfg.kalman_r_vis,
            r_enc=cfg.kalman_r_enc,
        )

    @staticmethod
    def initial_axis(e_deg: float = 0.0, theta_g_deg: float = 0.0) -> AxisKalmanState:
        """Zero-rate initial state at a given error and gimbal angle."""
        x = np.array([e_deg, theta_g_deg, 0.0, 0.0], dtype=np.float64)
        p = np.eye(4, dtype=np.float64)
        return AxisKalmanState(x=x, p=p)

    @staticmethod
    def initial_state(e_az_deg: float = 0.0, e_el_deg: float = 0.0) -> DualKalmanState:
        """Zero-rate dual-axis initial state."""
        return DualKalmanState(
            az=KalmanFilter.initial_axis(e_az_deg, 0.0),
            el=KalmanFilter.initial_axis(e_el_deg, 0.0),
        )


def predict(kf: KalmanFilter, state: AxisKalmanState, u: float, dt: float) -> AxisKalmanState:
    """Propagate one axis by dt under a held rate command u (deg/s)."""
    if dt <= 0.0:
        return state
    phi, b_d = discretize(kf.a_c, kf.b_c, dt)
    q_d = kf.q * dt
    x_pred = phi @ state.x + b_d.flatten() * u
    p_pred = phi @ state.p @ phi.T + q_d
    return AxisKalmanState(x=x_pred, p=p_pred)


def _update(
    state: AxisKalmanState,
    h: np.ndarray,
    r: float,
    z: float,
) -> Ok[AxisKalmanState] | Err[FaultCode]:
    """Scalar linear Kalman update. Err(GIMBAL_RUNAWAY) if S is singular."""
    s = float((h @ state.p @ h.T)[0, 0] + r)
    if abs(s) < 1e-18:
        return Err(FaultCode.GIMBAL_RUNAWAY)
    k = (state.p @ h.T) / s  # (4, 1)
    innovation = z - float((h @ state.x)[0])
    x_upd = state.x + k.flatten() * innovation
    p_upd = (np.eye(4, dtype=np.float64) - k @ h) @ state.p
    return Ok(AxisKalmanState(x=x_upd, p=p_upd))


def update_vis(
    kf: KalmanFilter, state: AxisKalmanState, z_vis: float
) -> Ok[AxisKalmanState] | Err[FaultCode]:
    """Incorporate a vision measurement of boresight error e (deg)."""
    return _update(state, kf.h_vis, kf.r_vis, z_vis)


def update_enc(
    kf: KalmanFilter, state: AxisKalmanState, theta_enc_deg: float
) -> Ok[AxisKalmanState] | Err[FaultCode]:
    """Incorporate an encoder measurement of gimbal angle theta_g (deg)."""
    return _update(state, kf.h_enc, kf.r_enc, theta_enc_deg)
