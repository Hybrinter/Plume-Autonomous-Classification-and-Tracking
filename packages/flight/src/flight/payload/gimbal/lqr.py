"""Discrete-time LQR controller for gimbal axis tracking.

Computes optimal control gains by solving the discrete algebraic Riccati
equation (DARE). Control law: u = -K * (x - x_desired)

Satisfies: REQ-GIMB-HIGH-001 (autonomous ROI retention),
           REQ-GIMB-HIGH-002 (stable bounded behavior).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from flight.libs.config import ControllerConfig


@dataclass(frozen=True)
class LqrController:
    """Discrete-time LQR with pre-computed gain matrix K (2x4)."""

    K: np.ndarray  # noqa: N815  np.ndarray[float64, (2,4)]
    max_slew_deg_s: float

    @staticmethod
    def from_config(cfg: ControllerConfig) -> LqrController:
        """Build LQR gain from ControllerConfig using DARE solver.

        System model: same constant-velocity model as KalmanFilter.
        A = F (4x4 state transition),
        B = [[0,0],[0,0],[dt,0],[0,dt]] (4x2 control input).
        """
        dt = cfg.kalman_dt_s
        A = np.array(  # noqa: N806
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        B = np.array(  # noqa: N806
            [
                [0, 0],
                [0, 0],
                [dt, 0],
                [0, dt],
            ],
            dtype=np.float64,
        )
        Q = np.diag(np.array(cfg.lqr_Q_diag, dtype=np.float64))  # noqa: N806
        R = np.diag(np.array(cfg.lqr_R_diag, dtype=np.float64))  # noqa: N806
        try:
            P = scipy.linalg.solve_discrete_are(A, B, Q, R)  # noqa: N806
            K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)  # noqa: N806
        except ValueError, np.linalg.LinAlgError:
            # Fallback to proportional control if DARE fails
            K = np.zeros((2, 4), dtype=np.float64)  # noqa: N806
            K[0, 0] = 1.0  # pan proportional
            K[1, 1] = 1.0  # tilt proportional
        return LqrController(K=K, max_slew_deg_s=cfg.max_slew_deg_s)


def compute_control(
    controller: LqrController,
    state_error: np.ndarray,  # np.ndarray[float64, (4,)] -- x_current - x_desired
) -> np.ndarray:  # np.ndarray[float64, (2,)] -- [pan_cmd_deg_s, tilt_cmd_deg_s]
    """Compute LQR control output: u = -K * error.

    Clamps output to max_slew_deg_s.
    """
    u = -controller.K @ state_error
    clamped: np.ndarray = np.clip(
        u,
        -controller.max_slew_deg_s,
        controller.max_slew_deg_s,
    )
    return clamped
