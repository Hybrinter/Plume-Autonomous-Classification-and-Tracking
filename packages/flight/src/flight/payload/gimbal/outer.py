"""Elevation rate-law primitives: smear cap, clips, and RateDecision (pure, SI).

Callers compose the scene and relative terms; rate_decision applies the
hardware slew clip, the stopping governor, and the science-window guards and
records the limit flags. r = sat(omega_scene + clip(relative; omega_sharp);
omega_hw). Azimuth rate stays on LosPrediction and pointing telemetry; it is not
an input here and does not shrink the elevation smear cap. rate_decision returns
a RateDecision; commanded_rate_rad_s is the absolute gimbal rate for set_rate.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001, REQ-GIMB-HIGH-003.
"""

from __future__ import annotations

# stdlib
import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RateDecision:
    """One outer-rate evaluation: requested terms, commanded rate, and limit flags.

    Attributes:
        scene_rate_rad_s: Scene-match rate used in this tick, rad/s.
        requested_relative_rate_rad_s: Smear-capped relative term before hardware
            and science limits, rad/s.
        requested_rate_rad_s: Absolute rate passed into finish, rad/s.
        commanded_rate_rad_s: Rate after hardware slew, stopping governor, and
            science-window guards, rad/s.
        smear_limit_rad_s: Elevation smear cap omega_sharp, rad/s.
        hardware_limited: True when hardware slew changed the requested rate.
        science_limited: True when the stopping governor or science-window guard
            changed the hardware-clipped rate.
    """

    scene_rate_rad_s: float
    requested_relative_rate_rad_s: float
    requested_rate_rad_s: float
    commanded_rate_rad_s: float
    smear_limit_rad_s: float
    hardware_limited: bool
    science_limited: bool


def smear_cap_rad_s(
    exposure_us: float,
    max_motion_smear_px: float,
    ifov_band_deg_per_px: float,
) -> float:
    """Elevation smear-limited |omega_rel| cap in rad/s.

    Inputs:
        exposure_us: Live frame exposure, microseconds. Non-positive yields +inf.
        max_motion_smear_px: Allowed along-track smear in upsampled band-plane pixels.
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


def stopping_cap(
    remaining_rad: float,
    max_decel_rad_s2: float,
    rate_loop_bandwidth_s: float,
) -> float:
    """Finite speed cap from remaining angle. Zero remaining is a zero cap.

    Inputs:
        remaining_rad: Angle left to the approached science bound, rad.
        max_decel_rad_s2: Stopping-distance deceleration, rad/s². Inf skips.
        rate_loop_bandwidth_s: Rate-loop bandwidth, 1/s. Inf skips.

    Outputs:
        float: Non-negative speed cap, rad/s. math.inf when both limits are
            infinite and remaining is positive.

    Notes:
        remaining <= 0 is returned as 0.0 before any infinite stopping product.
        Production rate mode passes infinite limits. The detailed plant passes
        finite deceleration and bandwidth.
    """
    remaining = max(0.0, remaining_rad)
    if remaining <= 0.0:
        return 0.0
    cap = math.inf
    if math.isfinite(max_decel_rad_s2):
        cap = min(cap, math.sqrt(2.0 * max_decel_rad_s2 * remaining))
    if math.isfinite(rate_loop_bandwidth_s):
        cap = min(cap, rate_loop_bandwidth_s * remaining)
    return cap


def boundary_limited(
    rate_rad_s: float,
    theta_g_rad: float,
    theta_sci_min_rad: float,
    theta_sci_max_rad: float,
    max_decel_rad_s2: float,
    rate_loop_bandwidth_s: float,
) -> float:
    """Limit commanded speed to the stopping distance inside the science window.

    Inputs:
        rate_rad_s: Rate after the hardware slew clip, rad/s.
        theta_g_rad: Current elevation, rad.
        theta_sci_min_rad: Science-window lower bound, rad.
        theta_sci_max_rad: Science-window upper bound, rad.
        max_decel_rad_s2: Stopping-distance deceleration, rad/s².
        rate_loop_bandwidth_s: Rate-loop bandwidth, 1/s.

    Outputs:
        float: Rate clipped by the approached-bound stopping cap, rad/s.
    """
    if rate_rad_s > 0.0:
        return min(
            rate_rad_s,
            stopping_cap(
                theta_sci_max_rad - theta_g_rad,
                max_decel_rad_s2,
                rate_loop_bandwidth_s,
            ),
        )
    if rate_rad_s < 0.0:
        return max(
            rate_rad_s,
            -stopping_cap(
                theta_g_rad - theta_sci_min_rad,
                max_decel_rad_s2,
                rate_loop_bandwidth_s,
            ),
        )
    return 0.0


def finish(
    rate_rad_s: float,
    theta_g_rad: float,
    theta_sci_min_rad: float,
    theta_sci_max_rad: float,
    omega_hw_rad_s: float,
    max_decel_rad_s2: float,
    rate_loop_bandwidth_s: float,
) -> float:
    """Apply hardware slew, the stopping governor, and science-window guards.

    Inputs:
        rate_rad_s: Requested absolute elevation rate, rad/s.
        theta_g_rad: Current elevation, rad.
        theta_sci_min_rad: Science-window lower bound, rad.
        theta_sci_max_rad: Science-window upper bound, rad.
        omega_hw_rad_s: Hardware slew cap, rad/s.
        max_decel_rad_s2: Stopping-distance deceleration, rad/s².
        rate_loop_bandwidth_s: Rate-loop bandwidth, 1/s.

    Outputs:
        float: Commanded absolute elevation rate, rad/s.

    Notes:
        TRACKING and REWIND share this limiter. Outward rate at either science
        bound is 0.0.
    """
    limited = boundary_limited(
        clip_rate(rate_rad_s, omega_hw_rad_s),
        theta_g_rad,
        theta_sci_min_rad,
        theta_sci_max_rad,
        max_decel_rad_s2,
        rate_loop_bandwidth_s,
    )
    if theta_g_rad <= theta_sci_min_rad + 1e-9 and limited < 0.0:
        return 0.0
    if theta_g_rad >= theta_sci_max_rad - 1e-9 and limited > 0.0:
        return 0.0
    return limited


def rate_decision(
    scene_rate_rad_s: float,
    requested_relative_rate_rad_s: float,
    requested_rate_rad_s: float,
    smear_limit_rad_s: float,
    theta_g_rad: float,
    theta_sci_min_rad: float,
    theta_sci_max_rad: float,
    omega_hw_rad_s: float,
    max_decel_rad_s2: float,
    rate_loop_bandwidth_s: float,
) -> RateDecision:
    """Run finish and record hardware and science limit flags.

    Inputs:
        scene_rate_rad_s: Scene-match rate, rad/s.
        requested_relative_rate_rad_s: Relative term before finish, rad/s.
        requested_rate_rad_s: Absolute rate passed into finish, rad/s.
        smear_limit_rad_s: Elevation smear cap, rad/s.
        theta_g_rad: Current elevation, rad.
        theta_sci_min_rad: Science-window lower bound, rad.
        theta_sci_max_rad: Science-window upper bound, rad.
        omega_hw_rad_s: Hardware slew cap, rad/s.
        max_decel_rad_s2: Stopping-distance deceleration, rad/s².
        rate_loop_bandwidth_s: Rate-loop bandwidth, 1/s.

    Outputs:
        RateDecision: Requested terms, commanded rate, and limit flags.
    """
    hw_clipped = clip_rate(requested_rate_rad_s, omega_hw_rad_s)
    commanded = finish(
        requested_rate_rad_s,
        theta_g_rad,
        theta_sci_min_rad,
        theta_sci_max_rad,
        omega_hw_rad_s,
        max_decel_rad_s2,
        rate_loop_bandwidth_s,
    )
    return RateDecision(
        scene_rate_rad_s=scene_rate_rad_s,
        requested_relative_rate_rad_s=requested_relative_rate_rad_s,
        requested_rate_rad_s=requested_rate_rad_s,
        commanded_rate_rad_s=commanded,
        smear_limit_rad_s=smear_limit_rad_s,
        hardware_limited=hw_clipped != requested_rate_rad_s,
        science_limited=commanded != hw_clipped,
    )
