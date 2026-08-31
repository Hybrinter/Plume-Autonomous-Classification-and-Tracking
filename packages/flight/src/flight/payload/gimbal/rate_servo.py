"""Two-axis rate PI with computed-torque decoupling (pure).

The servo tracks a rate reference r (rad/s) using encoder position. Measured rate
y_m is a differentiated, low-pass-filtered encoder signal. Two SISO PIs produce a
desired angular acceleration v. Torque is tau = J v + B y_m. Integrators freeze
when torque clips or a travel stop is active.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-002.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True, slots=True)
class RateServoState:
    """Immutable inner-loop servo state (SI units).

    Attributes:
        integral_az: Rate-error integral on azimuth (rad).
        integral_el: Rate-error integral on elevation (rad).
        last_theta_az_rad: Previous encoder azimuth (rad), or None before the first sample.
        last_theta_el_rad: Previous encoder elevation (rad), or None before the first sample.
        ym_az: Low-pass-filtered azimuth rate (rad/s).
        ym_el: Low-pass-filtered elevation rate (rad/s).
    """

    integral_az: float
    integral_el: float
    last_theta_az_rad: float | None
    last_theta_el_rad: float | None
    ym_az: float
    ym_el: float


INITIAL_RATE_SERVO_STATE = RateServoState(
    integral_az=0.0,
    integral_el=0.0,
    last_theta_az_rad=None,
    last_theta_el_rad=None,
    ym_az=0.0,
    ym_el=0.0,
)


def reset_servo(_state: RateServoState | None = None) -> RateServoState:
    """Return a zeroed servo state (clears the PI integrator)."""
    del _state
    return INITIAL_RATE_SERVO_STATE


def step(
    state: RateServoState,
    r_rad_s: tuple[float, float],
    theta_enc_rad: tuple[float, float],
    dt: float,
    j: np.ndarray,  # np.ndarray[float64, (2, 2)]
    b: np.ndarray,  # np.ndarray[float64, (2, 2)]
    kp: float,
    ki: float,
    tau_max_nm: float,
    ym_lpf_s: float,
    travel_saturated: tuple[bool, bool] = (False, False),
) -> tuple[RateServoState, tuple[float, float]]:
    """Advance the rate PI one tick and return (new_state, (tau_az, tau_el) in N·m).

    Inputs:
        state: Previous servo state.
        r_rad_s: Rate reference (azimuth, elevation) in rad/s.
        theta_enc_rad: Encoder angles (azimuth, elevation) in rad.
        dt: Tick duration in seconds. Values <= 0 return zero torque and the input state.
        j: Inertia estimate, shape (2, 2), kg m^2.
        b: Damping estimate, shape (2, 2), N m s / rad.
        kp: Proportional gain on rate error (1/s).
        ki: Integral gain on rate error (1/s^2).
        tau_max_nm: Symmetric torque clip per axis (N·m).
        ym_lpf_s: First-order time constant for encoder-rate filtering (s). Must be > 0.
        travel_saturated: True per axis when that axis is against a travel stop.

    Outputs:
        (new_state, (tau_az_nm, tau_el_nm)).
    """
    if dt <= 0.0:
        return state, (0.0, 0.0)

    theta_az, theta_el = theta_enc_rad
    y_raw_az = (
        (theta_az - state.last_theta_az_rad) / dt if state.last_theta_az_rad is not None else 0.0
    )
    y_raw_el = (
        (theta_el - state.last_theta_el_rad) / dt if state.last_theta_el_rad is not None else 0.0
    )
    lpf_alpha = dt / (ym_lpf_s + dt)
    ym_az = state.ym_az + lpf_alpha * (y_raw_az - state.ym_az)
    ym_el = state.ym_el + lpf_alpha * (y_raw_el - state.ym_el)

    e_az = r_rad_s[0] - ym_az
    e_el = r_rad_s[1] - ym_el
    i_az = state.integral_az
    i_el = state.integral_el
    if not travel_saturated[0]:
        i_az = state.integral_az + e_az * dt
    if not travel_saturated[1]:
        i_el = state.integral_el + e_el * dt

    v = np.array(
        [kp * e_az + ki * i_az, kp * e_el + ki * i_el],
        dtype=np.float64,
    )
    if travel_saturated[0]:
        v[0] = 0.0
    if travel_saturated[1]:
        v[1] = 0.0
    ym = np.array([ym_az, ym_el], dtype=np.float64)
    tau = j @ v + b @ ym
    tau_clip = np.clip(tau, -tau_max_nm, tau_max_nm)

    if float(abs(tau[0])) > tau_max_nm:
        i_az = state.integral_az
    if float(abs(tau[1])) > tau_max_nm:
        i_el = state.integral_el

    new_state = replace(
        state,
        integral_az=i_az,
        integral_el=i_el,
        last_theta_az_rad=theta_az,
        last_theta_el_rad=theta_el,
        ym_az=ym_az,
        ym_el=ym_el,
    )
    return new_state, (float(tau_clip[0]), float(tau_clip[1]))
