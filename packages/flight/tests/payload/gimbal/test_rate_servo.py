"""Unit tests for the pure inner rate servo (PI + computed torque)."""

from __future__ import annotations

import numpy as np
from flight.payload.gimbal.rate_servo import INITIAL_RATE_SERVO_STATE, reset_servo, step

_J_DIAG = np.diag([1.0, 1.0])
_B_ZERO = np.zeros((2, 2))
_DT = 0.005
_KP = 20.0
_KI = 40.0
_TAU_MAX = 50.0
_LPF = 0.02


def _plant_step(
    theta: np.ndarray,
    omega: np.ndarray,
    tau: tuple[float, float],
    j: np.ndarray,
    b: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One Euler step of J ω̇ + B ω = τ."""
    omega_dot = np.linalg.inv(j) @ (np.array(tau, dtype=np.float64) - b @ omega)
    omega_next = omega + omega_dot * dt
    theta_next = theta + omega_next * dt
    return theta_next, omega_next


def test_dt_non_positive_returns_zero_torque() -> None:
    """A non-positive dt yields zero torque and leaves integrals at zero."""
    state, tau = step(
        INITIAL_RATE_SERVO_STATE,
        (0.1, 0.0),
        (0.0, 0.0),
        0.0,
        _J_DIAG,
        _B_ZERO,
        _KP,
        _KI,
        _TAU_MAX,
        _LPF,
    )
    assert tau == (0.0, 0.0)
    assert state.integral_az == 0.0
    assert state.integral_el == 0.0


def test_reset_clears_integrator() -> None:
    """reset_servo returns the zero initial state."""
    state, _tau = step(
        INITIAL_RATE_SERVO_STATE,
        (1.0, 0.0),
        (0.0, 0.0),
        _DT,
        _J_DIAG,
        _B_ZERO,
        _KP,
        _KI,
        _TAU_MAX,
        _LPF,
    )
    assert state.integral_az != 0.0 or state.ym_az != 0.0 or state.last_theta_az_rad is not None
    cleared = reset_servo(state)
    assert cleared == INITIAL_RATE_SERVO_STATE


def test_step_response_tracks_rate_reference() -> None:
    """Closed-loop rate on a diagonal plant approaches a constant azimuth reference."""
    j = _J_DIAG
    b = np.diag([0.2, 0.2])
    state = INITIAL_RATE_SERVO_STATE
    theta = np.zeros(2)
    omega = np.zeros(2)
    r = (0.2, 0.0)
    for _ in range(400):
        state, tau = step(
            state, r, (float(theta[0]), float(theta[1])), _DT, j, b, _KP, _KI, _TAU_MAX, _LPF
        )
        theta, omega = _plant_step(theta, omega, tau, j, b, _DT)
    assert abs(omega[0] - r[0]) < 0.03
    assert abs(omega[1]) < 0.03


def test_diagonal_j_keeps_idle_axis_near_zero_torque() -> None:
    """With diagonal J, an azimuth-only reference produces near-zero elevation torque."""
    state, tau = step(
        INITIAL_RATE_SERVO_STATE,
        (0.5, 0.0),
        (0.0, 0.0),
        _DT,
        _J_DIAG,
        _B_ZERO,
        _KP,
        _KI,
        _TAU_MAX,
        _LPF,
    )
    assert abs(tau[0]) > abs(tau[1])
    assert abs(tau[1]) < 1e-12


def test_full_j_decouples_when_inertia_matches() -> None:
    """Computed torque with the true J keeps elevation rate small on an azimuth command."""
    j = np.array([[1.0, 0.3], [0.3, 1.0]], dtype=np.float64)
    b = np.diag([0.1, 0.1])
    state = INITIAL_RATE_SERVO_STATE
    theta = np.zeros(2)
    omega = np.zeros(2)
    r = (0.15, 0.0)
    for _ in range(400):
        state, tau = step(
            state, r, (float(theta[0]), float(theta[1])), _DT, j, b, _KP, _KI, _TAU_MAX, _LPF
        )
        theta, omega = _plant_step(theta, omega, tau, j, b, _DT)
    assert abs(omega[0] - r[0]) < 0.04
    assert abs(omega[1]) < 0.04


def test_torque_clip_stops_integral_growth() -> None:
    """When torque saturates, the azimuth integral does not keep growing."""
    state = INITIAL_RATE_SERVO_STATE
    theta = 0.0
    integrals: list[float] = []
    for _ in range(50):
        state, tau = step(
            state,
            (10.0, 0.0),
            (theta, 0.0),
            _DT,
            _J_DIAG,
            _B_ZERO,
            _KP,
            _KI,
            0.05,
            _LPF,
        )
        assert abs(tau[0]) <= 0.05 + 1e-12
        integrals.append(state.integral_az)
        theta += 0.0
    assert integrals[-1] <= integrals[10] + 1e-9


def test_travel_stop_zeros_axis_command() -> None:
    """A saturated travel flag zeros that axis PI output."""
    primed, _ = step(
        INITIAL_RATE_SERVO_STATE,
        (0.5, 0.0),
        (0.0, 0.0),
        _DT,
        _J_DIAG,
        _B_ZERO,
        _KP,
        _KI,
        _TAU_MAX,
        _LPF,
    )
    _state, tau = step(
        primed,
        (0.5, 0.0),
        (0.01, 0.0),
        _DT,
        _J_DIAG,
        _B_ZERO,
        _KP,
        _KI,
        _TAU_MAX,
        _LPF,
        travel_saturated=(True, False),
    )
    assert tau[0] == 0.0
