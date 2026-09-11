"""Tracking elevation rate law: scene match plus smear-capped relative rate (pure, SI).

r = sat(omega_scene + clip(relative; omega_sharp); omega_hw). TRACKING matches
omega_t_nom + omega_t_res and smear-caps only K_p * e_hat. REWIND matches the
boresight-ground nom and smear-caps the hunt, then escapes to the hardware cap
after rewind_sharp_max_s. omega_az is diagnostic and does not shrink the
elevation smear cap. The returned r is an absolute gimbal rate for set_rate.

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
    """Elevation smear-limited |omega_rel| cap in rad/s.

    Inputs:
        exposure_us: Live frame exposure, microseconds. Non-positive yields +inf.
        max_motion_smear_px: Allowed along-track smear in band-plane pixels.
        ifov_band_deg_per_px: Band-plane IFOV in degrees per pixel.

    Outputs:
        float: omega_sharp,el in rad/s. math.inf when exposure is not positive.

    Notes:
        This is the full elevation budget. Unactuated azimuth rate does not
        reduce it.
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
    if not math.isfinite(limit_rad_s):
        return r_rad_s
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
    rewind_elapsed_s: float = 0.0,
    rewind_sharp_max_s: float = 2.0,
    omega_az: float = 0.0,
) -> float:
    """Compute the absolute elevation rate reference r in rad/s.

    Inputs:
        omega_t_nom: Co-rotating predictor elevation rate, rad/s.
        omega_t_res: Residual-rate estimate, rad/s. Ignored in REWIND.
        e_hat: Residual-filter elevation error, rad.
        k_p: Outer proportional gain, 1/s.
        mode: Arbiter mode.
        live: True when the arbiter has a current target this tick.
        theta_g_rad: Current elevation, rad.
        theta_sci_max_rad: Science-limb elevation, rad.
        omega_hw_rad_s: Hardware slew cap, rad/s.
        exposure_us: Live (or last) exposure for the elevation smear cap.
        max_motion_smear_px: Along-track smear budget, pixels.
        ifov_band_deg_per_px: Band-plane IFOV, deg/px.
        theta_sci_min_rad: Science-window lower bound, rad.
        max_decel_rad_s2: Stopping-distance deceleration, rad/s².
        rate_loop_bandwidth_s: Rate-loop bandwidth used in the stopping governor.
        rewind_elapsed_s: Time in REWIND, seconds.
        rewind_sharp_max_s: Sharp REWIND window before hardware-slew escape.
        omega_az: Unactuated azimuth rate, rad/s. Diagnostic only.

    Outputs:
        float: Absolute gimbal rate reference r, rad/s.
    """
    del omega_az
    omega_sharp = smear_cap_rad_s(exposure_us, max_motion_smear_px, ifov_band_deg_per_px)
    rate_cap = omega_hw_rad_s

    def stopping_cap(remaining_rad: float) -> float:
        """Finite speed cap from remaining angle. Zero remaining is a zero cap."""
        remaining = max(0.0, remaining_rad)
        if remaining <= 0.0:
            return 0.0
        cap = math.inf
        if math.isfinite(max_decel_rad_s2):
            cap = min(cap, math.sqrt(2.0 * max_decel_rad_s2 * remaining))
        if math.isfinite(rate_loop_bandwidth_s):
            cap = min(cap, rate_loop_bandwidth_s * remaining)
        return cap

    def boundary_limited(rate_rad_s: float) -> float:
        """Limit commanded speed to the stopping distance inside the science window."""
        if rate_rad_s > 0.0:
            return min(rate_rad_s, stopping_cap(theta_sci_max_rad - theta_g_rad))
        if rate_rad_s < 0.0:
            return max(rate_rad_s, -stopping_cap(theta_g_rad - theta_sci_min_rad))
        return 0.0

    def finish(rate_rad_s: float) -> float:
        """Apply hardware slew, the stopping governor, and science-window guards."""
        limited = boundary_limited(clip_rate(rate_rad_s, rate_cap))
        if theta_g_rad <= theta_sci_min_rad + 1e-9 and limited < 0.0:
            return 0.0
        if theta_g_rad >= theta_sci_max_rad - 1e-9 and limited > 0.0:
            return 0.0
        return limited

    if mode is GimbalState.REWIND:
        if theta_g_rad >= theta_sci_max_rad - 1e-9:
            return 0.0
        direction = 1.0 if (theta_sci_max_rad - theta_g_rad) >= 0.0 else -1.0
        if rewind_elapsed_s >= rewind_sharp_max_s:
            return finish(direction * rate_cap)
        omega_rel = clip_rate(direction * omega_sharp, omega_sharp)
        return finish(omega_t_nom + omega_rel)

    if mode is GimbalState.TRACKING and live:
        omega_scene = omega_t_nom + omega_t_res
        omega_rel = clip_rate(k_p * e_hat, omega_sharp)
        return finish(omega_scene + omega_rel)

    return 0.0
