"""Outer elevation rate law: feedforward + residual with hardware clips (pure, SI).

r = sat(omega_t_nom + omega_t_res + K_p * e_hat; omega_hw). Image-smear
estimates qualify science frames separately and do not reduce control authority.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001, REQ-GIMB-HIGH-003.
"""

from __future__ import annotations

# stdlib
import math

from flight.libs.types import GimbalState


def smear_cap_rad_s(
    exposure_us: float,
    max_motion_smear_px: float,
    ifov_band_deg_per_px: float,
) -> float:
    """Live smear-limited |r| cap in rad/s.

    Inputs:
        exposure_us: Live frame exposure, microseconds. Non-positive yields +inf.
        max_motion_smear_px: Allowed smear in band-plane pixels.
        ifov_band_deg_per_px: Band-plane IFOV in degrees per pixel.

    Outputs:
        float: r_max,img in rad/s. math.inf when exposure is not positive.
    """
    dt_exp_s = exposure_us * 1.0e-6
    if dt_exp_s <= 0.0:
        return math.inf
    ifov_rad = math.radians(ifov_band_deg_per_px)
    return (max_motion_smear_px * ifov_rad) / dt_exp_s


def clip_rate(r_rad_s: float, limit_rad_s: float) -> float:
    """Clip a rate to +-limit.

    Inputs:
        r_rad_s: Unsaturated rate, rad/s.
        limit_rad_s: Symmetric bound, rad/s. Non-positive yields 0.0.

    Outputs:
        float: Clipped rate.
    """
    if limit_rad_s <= 0.0:
        return 0.0
    if r_rad_s > limit_rad_s:
        return limit_rad_s
    if r_rad_s < -limit_rad_s:
        return -limit_rad_s
    return r_rad_s


def outer_rate(
    omega_t_nom: float,
    omega_t_res: float,
    e_hat: float,
    k_p: float,
    mode: GimbalState,
    live: bool,
    theta_g_rad: float,
    theta_sci_max_rad: float,
    omega_hw_rad_s: float,
    exposure_us: float,
    max_motion_smear_px: float,
    ifov_band_deg_per_px: float,
    theta_sci_min_rad: float = 0.0,
    max_decel_rad_s2: float = math.inf,
    rate_loop_bandwidth_s: float = math.inf,
) -> float:
    """Compute the outer rate reference r in rad/s.

    Inputs:
        omega_t_nom: Co-rotating predictor rate, rad/s.
        omega_t_res: Residual-rate estimate, rad/s.
        e_hat: Residual-filter elevation error, rad.
        k_p: Outer proportional gain, 1/s.
        mode: Arbiter mode.
        live: True when the arbiter has a current target this tick.
        theta_g_rad: Current elevation, rad.
        theta_sci_max_rad: Science-limb elevation, rad.
        theta_sci_min_rad: Science-window lower bound, rad.
        omega_hw_rad_s: Hardware slew cap, rad/s.
        exposure_us: Live (or last) exposure for the smear cap.
        max_motion_smear_px: Smear budget, pixels.
        ifov_band_deg_per_px: Band-plane IFOV, deg/px.

    Outputs:
        float: Rate reference r, rad/s.
    """
    # Retain the optical arguments while callers migrate to an explicit science
    # qualification result. They must not silently change control authority.
    del exposure_us, max_motion_smear_px, ifov_band_deg_per_px
    rate_cap = omega_hw_rad_s

    def boundary_limited(rate_rad_s: float) -> float:
        """Limit commanded speed to the stopping distance inside the science window."""
        if rate_rad_s > 0.0:
            remaining = max(0.0, theta_sci_max_rad - theta_g_rad)
            stopping_cap = math.sqrt(2.0 * max_decel_rad_s2 * remaining)
            stopping_cap = min(stopping_cap, rate_loop_bandwidth_s * remaining)
            return min(rate_rad_s, stopping_cap)
        if rate_rad_s < 0.0:
            remaining = max(0.0, theta_g_rad - theta_sci_min_rad)
            stopping_cap = math.sqrt(2.0 * max_decel_rad_s2 * remaining)
            stopping_cap = min(stopping_cap, rate_loop_bandwidth_s * remaining)
            return max(rate_rad_s, -stopping_cap)
        return 0.0

    if mode is GimbalState.REWIND:
        if theta_g_rad >= theta_sci_max_rad - 1e-9:
            return 0.0
        direction = 1.0 if (theta_sci_max_rad - theta_g_rad) >= 0.0 else -1.0
        return boundary_limited(clip_rate(direction * rate_cap, rate_cap))

    if mode is GimbalState.TRACKING and live:
        r = omega_t_nom + omega_t_res + k_p * e_hat
        r = clip_rate(r, rate_cap)
        if theta_g_rad <= theta_sci_min_rad + 1e-9 and r < 0.0:
            return 0.0
        if theta_g_rad >= theta_sci_max_rad - 1e-9 and r > 0.0:
            return 0.0
        return boundary_limited(r)

    return 0.0
