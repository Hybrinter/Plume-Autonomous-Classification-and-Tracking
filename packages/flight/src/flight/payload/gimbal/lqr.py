"""Continuous-time LQR rate command for one gimbal axis (pure).

The Kalman filter estimates [e, theta_g, omega_t, omega_g]. Target rate omega_t is
not controllable, so the LQR design plant is the stabilizable pair [e, omega_g]
with e_dot = -omega_g (design model takes omega_t = 0) and
omega_g_dot = (-omega_g + u) / tau_cl. u = -K [e, omega_g] is the physical
inner-loop rate reference in deg/s.

Satisfies: REQ-GIMB-HIGH-001, REQ-GIMB-HIGH-002.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from flight.libs.config import ControllerConfig


def lqr_plant(tau_cl_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (A_c, B_c) for x = [e, omega_g], u = rate command (deg/s)."""
    inv_tau = 1.0 / tau_cl_s
    a_c = np.array([[0.0, -1.0], [0.0, -inv_tau]], dtype=np.float64)
    b_c = np.array([[0.0], [inv_tau]], dtype=np.float64)
    return a_c, b_c


@dataclass(frozen=True, slots=True)
class LqrController:
    """Continuous LQR with gain K (1x2) on [e, omega_g] and a slew clamp (deg/s)."""

    k: np.ndarray  # np.ndarray[float64, (1, 2)]
    max_slew_deg_s: float

    @staticmethod
    def from_config(cfg: ControllerConfig) -> LqrController:
        """Solve CARE on the [e, omega_g] plant and build K.

        Q uses lqr_Q_diag[0] on e and lqr_Q_diag[3] on omega_g. R is the first
        entry of lqr_R_diag.
        """
        a_c, b_c = lqr_plant(cfg.tau_cl_s)
        q_diag = np.array(cfg.lqr_Q_diag, dtype=np.float64)
        q_e = float(q_diag[0]) if q_diag.size > 0 else 10.0
        q_wg = float(q_diag[3]) if q_diag.size > 3 else 1.0
        q = np.diag(np.array([q_e, q_wg], dtype=np.float64))
        r_val = float(cfg.lqr_R_diag[0]) if cfg.lqr_R_diag else 1.0
        r = np.array([[r_val]], dtype=np.float64)
        try:
            p = scipy.linalg.solve_continuous_are(a_c, b_c, q, r)
            k = np.linalg.inv(r) @ (b_c.T @ p)
        except ValueError, np.linalg.LinAlgError:
            k = np.zeros((1, 2), dtype=np.float64)
            k[0, 0] = 1.0
        return LqrController(k=k, max_slew_deg_s=cfg.max_slew_deg_s)


def compute_axis_control(controller: LqrController, x: np.ndarray) -> float:
    """Return the clamped rate command u = -K [e, omega_g] for one 4-state axis (deg/s)."""
    reduced = np.array([float(x[0]), float(x[3])], dtype=np.float64)
    u = float((-controller.k @ reduced).reshape(-1)[0])
    limit = controller.max_slew_deg_s
    return min(max(u, -limit), limit)


def compute_control(
    controller: LqrController,
    x_az: np.ndarray,
    x_el: np.ndarray,
) -> np.ndarray:
    """Return clamped [az_rate, el_rate] deg/s from the two axis 4-state vectors."""
    return np.array(
        [
            compute_axis_control(controller, x_az),
            compute_axis_control(controller, x_el),
        ],
        dtype=np.float64,
    )
