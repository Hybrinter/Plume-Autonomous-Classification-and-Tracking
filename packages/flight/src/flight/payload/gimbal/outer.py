"""Outer elevation rate law: feedforward + residual + smear / hardware clips (pure, SI).

r = sat(omega_t_nom + omega_t_res + K_p * e_hat; r_max(mode)). TRACKING (live) and
REWIND clip to the live smear cap and the hardware slew. SAFE / STOW / HOME use the
position loop and are not smear-capped.

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
) -> float:
    """Compute the outer rate reference r in rad/s.

    Inputs:
        omega_t_nom: Co-rotating predictor rate, rad/s.
        omega_t_res: Residual-rate estimate, rad/s.
        e_hat: Residual-filter elevation error, rad.
        k_p: Outer proportional gain, 1/s.
        mode: Arbiter mode.
        live: True after the first accepted vision update (TRACKING live).
        theta_g_rad: Current elevation, rad.
        theta_sci_max_rad: Science-limb elevation, rad.
        omega_hw_rad_s: Hardware slew cap, rad/s.
        exposure_us: Live (or last) exposure for the smear cap.
        max_motion_smear_px: Smear budget, pixels.
        ifov_band_deg_per_px: Band-plane IFOV, deg/px.

    Outputs:
        float: Rate reference r, rad/s.
    """
    r_smear = smear_cap_rad_s(exposure_us, max_motion_smear_px, ifov_band_deg_per_px)
    img_cap = min(r_smear, omega_hw_rad_s)

    if mode is GimbalState.REWIND:
        direction = 1.0 if (theta_sci_max_rad - theta_g_rad) >= 0.0 else -1.0
        return clip_rate(direction * img_cap, img_cap)

    if mode is GimbalState.TRACKING and live:
        r = omega_t_nom + omega_t_res + k_p * e_hat
        return clip_rate(r, img_cap)

    return 0.0
