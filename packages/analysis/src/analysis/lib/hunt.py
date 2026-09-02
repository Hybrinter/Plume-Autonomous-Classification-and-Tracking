"""Elevation-rewind and limb-scan hunt kinematics.

After track loss the gimbal slews elevation toward the science limb stop.
During that slew the look point races ahead of the ISS and the FOV ribbon
can acquire a new stack. After the limb stop, one-axis waits on orbital
motion through the FOV ribbon; two-axis rasters azimuth across the keep-out
box at the smear-limited scan rate.

Contains:
  - HuntResult / HuntModel.
  - two_phase_mean_wait_s / ground_central_angle_rad / look_ahead_km.
  - az_scan_rate_deg_s / reacquire.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from analysis.lib.constants import OMEGA_EARTH_RAD_S
from analysis.lib.look import GimbalBox, look_at, rotate_z
from analysis.lib.optics import Optics
from analysis.lib.orbit import (
    Orbit,
    argument_of_latitude,
    heading_from_north_deg,
    iss_eci,
    offset_ecef,
    origin_ecef,
)


@dataclass(frozen=True)
class HuntResult:
    """One-axis and two-axis mean reacquire times plus scan diagnostics.

    Attributes:
        t_reacq_1_s: Mean one-axis wait from loss to next acquire, seconds.
        t_reacq_2_s: Mean two-axis wait from loss to next acquire, seconds.
        t_rewind_1_s: Elevation rewind duration after one-axis loss, seconds.
        t_rewind_2_s: Elevation rewind duration after two-axis loss, seconds.
        omega_az_scan_deg_s: Smear-limited azimuth raster rate at the limb.
        scan_fill: Along-track fill fraction of the two-axis raster, in [0, 1].
        swath_fov_km: Instantaneous FOV ground width at the limb look, km.
        swath_box_km: Pointing-box ground width at the limb look, km.
    """

    t_reacq_1_s: float
    t_reacq_2_s: float
    t_rewind_1_s: float
    t_rewind_2_s: float
    omega_az_scan_deg_s: float
    scan_fill: float
    swath_fov_km: float
    swath_box_km: float


@dataclass(frozen=True)
class HuntModel:
    """Locked optics, orbit, box, and smear caps for hunt waits.

    Attributes:
        optics: Usable sensor FOV.
        orbit: Circular ISS orbit.
        box: Operational elevation window and azimuth keep-out.
        omega_img_deg_s: Imaging rewind cap (smear limit minus peak track).
        omega_rel_deg_s: Scene-relative smear limit, degrees per second.
    """

    optics: Optics
    orbit: Orbit
    box: GimbalBox
    omega_img_deg_s: float
    omega_rel_deg_s: float

    def wait(
        self,
        lat_deg: float,
        dens_per_km2: float,
        t_dwell_1_s: float,
        t_dwell_2_s: float,
        t_window_s: float,
    ) -> HuntResult:
        """Return reacquire times at signed latitude for both gimbals.

        Args:
            lat_deg: Geocentric latitude in degrees (signed).
            dens_per_km2: Stack density per square kilometre.
            t_dwell_1_s: One-axis single-target dwell, seconds.
            t_dwell_2_s: Two-axis single-target dwell, seconds.
            t_window_s: Science-window length, seconds.

        Returns:
            HuntResult with waits and scan diagnostics.
        """
        return reacquire(
            lat_deg,
            dens_per_km2,
            self.optics,
            self.orbit,
            self.box,
            t_dwell_1_s=t_dwell_1_s,
            t_dwell_2_s=t_dwell_2_s,
            t_window_s=t_window_s,
            omega_img_deg_s=self.omega_img_deg_s,
            omega_rel_deg_s=self.omega_rel_deg_s,
        )


def two_phase_mean_wait_s(lambda_a: float, t_a: float, lambda_b: float) -> float:
    """Return mean wait for rate ``lambda_a`` during ``t_a``, then ``lambda_b``.

    The waiting time is that of an inhomogeneous Poisson process: rewind
    search at constant rate lambda_a for duration t_a, then limb search at
    lambda_b forever. E[T] = integral of the survival function.

    Args:
        lambda_a: Encounter rate during the finite first phase, 1/s.
        t_a: First-phase duration in seconds.
        lambda_b: Encounter rate after t_a, 1/s.

    Returns:
        Mean wait in seconds, or inf if both rates are zero.
    """
    dur = max(0.0, t_a)
    if lambda_a <= 0.0 and lambda_b <= 0.0:
        return math.inf
    if lambda_a <= 0.0:
        return dur + 1.0 / lambda_b
    survive = math.exp(-lambda_a * dur)
    first = (1.0 - survive) / lambda_a
    if lambda_b <= 0.0:
        return math.inf if survive > 0.0 else first
    return first + survive / lambda_b


def ground_central_angle_rad(eta_deg: float, r_iss_km: float, r_earth_km: float) -> float:
    """Return Earth-central angle from sub-satellite point to the look point.

    Args:
        eta_deg: Off-nadir angle in degrees.
        r_iss_km: ISS geocentric radius in kilometres.
        r_earth_km: Earth geocentric radius in kilometres.

    Returns:
        Central angle in radians. Zero at nadir. Clipped at the limb.
    """
    eta = math.radians(max(0.0, eta_deg))
    if eta < 1e-12:
        return 0.0
    sine_arg = (r_iss_km / r_earth_km) * math.sin(eta)
    if sine_arg >= 1.0:
        return math.asin(r_earth_km / r_iss_km)  # unused; caller should not be at limb
    return math.asin(sine_arg) - eta


def look_ahead_km(eta_deg: float, r_iss_km: float, r_earth_km: float) -> float:
    """Return along-track ground range from nadir to the look point.

    Args:
        eta_deg: Off-nadir angle in degrees.
        r_iss_km: ISS geocentric radius in kilometres.
        r_earth_km: Earth geocentric radius in kilometres.

    Returns:
        Ground range in kilometres.
    """
    return r_earth_km * ground_central_angle_rad(eta_deg, r_iss_km, r_earth_km)


def d_look_ahead_d_eta_km_per_rad(eta_deg: float, r_iss_km: float, r_earth_km: float) -> float:
    """Return ds/d(eta) of look-ahead ground range, km per radian of off-nadir.

    Args:
        eta_deg: Off-nadir angle in degrees.
        r_iss_km: ISS geocentric radius in kilometres.
        r_earth_km: Earth geocentric radius in kilometres.

    Returns:
        Derivative in kilometres per radian. Zero at nadir; large near the limb.
    """
    eta = math.radians(max(0.0, eta_deg))
    if eta < 1e-6:
        h_km = r_iss_km - r_earth_km
        return r_iss_km * r_earth_km / max(h_km, 1e-6)
    sine_arg = (r_iss_km / r_earth_km) * math.sin(eta)
    if sine_arg >= 0.999:
        return 1.0e6
    d_delta = (r_iss_km / r_earth_km) * math.cos(eta) / math.sqrt(1.0 - sine_arg * sine_arg)
    return r_earth_km * (d_delta - 1.0)


def slant_at_eta_km(eta_deg: float, r_iss_km: float, r_earth_km: float) -> float:
    """Return slant range to the look point at off-nadir ``eta_deg``.

    Args:
        eta_deg: Off-nadir angle in degrees.
        r_iss_km: ISS geocentric radius in kilometres.
        r_earth_km: Earth geocentric radius in kilometres.

    Returns:
        Slant range in kilometres.
    """
    if eta_deg < 1e-6:
        return r_iss_km - r_earth_km
    delta = ground_central_angle_rad(eta_deg, r_iss_km, r_earth_km)
    sine_eta = math.sin(math.radians(eta_deg))
    if abs(sine_eta) < 1e-12:
        return r_iss_km - r_earth_km
    return r_iss_km * math.sin(delta) / sine_eta


def cross_track_width_km(half_az_deg: float, slant_km: float) -> float:
    """Return full cross-track ground width of a half-angle at one slant.

    Args:
        half_az_deg: Half-angle in degrees.
        slant_km: Slant range in kilometres.

    Returns:
        Full width in kilometres.
    """
    return 2.0 * slant_km * math.tan(math.radians(half_az_deg))


def az_scan_rate_deg_s(omega_rel_deg_s: float, omega_el_scene_deg_s: float) -> float:
    """Return smear-limited azimuth raster rate while holding elevation.

    Total scene-relative rate is hypot(az_slew, el_scene). The imaging gate
    caps that hypot, so leftover budget after the along-track scene rate
    is the max az slew.

    Args:
        omega_rel_deg_s: Scene-relative smear limit, degrees per second.
        omega_el_scene_deg_s: |d(el)/dt| of a ground point at the look, deg/s.

    Returns:
        Non-negative azimuth slew cap in degrees per second.
    """
    leftover_sq = omega_rel_deg_s * omega_rel_deg_s - omega_el_scene_deg_s * omega_el_scene_deg_s
    if leftover_sq <= 0.0:
        return 0.0
    return math.sqrt(leftover_sq)


def _el_loss_deg(box: GimbalBox, t_dwell_s: float, t_window_s: float) -> float:
    """Return elevation at track loss, interpolating limb -> nadir over the window.

    Args:
        box: Gimbal box.
        t_dwell_s: In-frame dwell, seconds.
        t_window_s: Science-window length, seconds.

    Returns:
        Elevation in degrees, clipped to [el_limb, el_nadir].
    """
    if t_window_s <= 1e-9:
        return box.el_limb_deg
    frac = min(1.0, max(0.0, t_dwell_s / t_window_s))
    return box.el_limb_deg + frac * (box.el_nadir_deg - box.el_limb_deg)


def _rewind_time_s(el_loss_deg: float, box: GimbalBox, omega_img_deg_s: float) -> float:
    """Return time to slew from ``el_loss_deg`` down to the limb stop.

    Args:
        el_loss_deg: Elevation at loss, degrees.
        box: Gimbal box.
        omega_img_deg_s: Imaging rewind cap, degrees per second.

    Returns:
        Rewind duration in seconds. Zero if already at the stop.
    """
    delta = max(0.0, el_loss_deg - box.el_limb_deg)
    if omega_img_deg_s <= 1e-12:
        return math.inf if delta > 0.0 else 0.0
    return delta / omega_img_deg_s


def scene_el_rate_deg_s(orbit: Orbit, box: GimbalBox, lat_deg: float, eta_deg: float) -> float:
    """Return |d(el)/dt| of a ground point currently at off-nadir ``eta_deg``.

    Finite-differences two looks 50 ms apart with Earth rotation on.

    Args:
        orbit: Circular ISS orbit.
        box: Gimbal box (nadir elevation convention).
        lat_deg: Pass geocentric latitude in degrees.
        eta_deg: Off-nadir angle of the look, degrees.

    Returns:
        Absolute elevation rate in degrees per second.
    """
    r_earth = orbit.earth_radius_km(lat_deg)
    ahead = look_ahead_km(eta_deg, orbit.radius_km, r_earth)
    u0 = argument_of_latitude(lat_deg, orbit.inclination_rad)
    heading = heading_from_north_deg(lat_deg, orbit.inclination_rad)
    origin = origin_ecef(orbit, lat_deg)
    tgt_ecef = offset_ecef(origin, lat_deg, ahead, 0.0, heading)
    dt_s = 0.05

    def _el(t_s: float) -> float:
        r_iss, vel = iss_eci(t_s, orbit, u0)
        tgt = rotate_z(OMEGA_EARTH_RAD_S * t_s) @ tgt_ecef
        return look_at(r_iss, vel, tgt, box.el_nadir_deg).el_deg

    return abs(_el(dt_s) - _el(0.0)) / dt_s


def reacquire(
    lat_deg: float,
    dens_per_km2: float,
    optics: Optics,
    orbit: Orbit,
    box: GimbalBox,
    *,
    t_dwell_1_s: float,
    t_dwell_2_s: float,
    t_window_s: float,
    omega_img_deg_s: float,
    omega_rel_deg_s: float,
) -> HuntResult:
    """Return one-axis and two-axis mean reacquire times at ``lat_deg``.

    Rewind search uses the FOV ribbon for both gimbals (no azimuth raster
    until the limb stop). Limb search uses the FOV ribbon for one-axis and
    a smear-gated +/-az_box raster for two-axis. Look-point speed during
    rewind is ISS ground speed plus ds/d(eta) * rewind rate.

    Args:
        lat_deg: Geocentric latitude in degrees (signed).
        dens_per_km2: Stack density per square kilometre.
        optics: Usable sensor FOV.
        orbit: Circular ISS orbit.
        box: Elevation window and azimuth keep-out.
        t_dwell_1_s: One-axis dwell, seconds.
        t_dwell_2_s: Two-axis dwell, seconds.
        t_window_s: Science-window length, seconds.
        omega_img_deg_s: Imaging rewind cap, degrees per second.
        omega_rel_deg_s: Scene-relative smear limit, degrees per second.

    Returns:
        HuntResult. Infinite waits when density or speed is zero.
    """
    lat_abs = abs(float(lat_deg))
    r_earth = orbit.earth_radius_km(lat_abs)
    r_iss = orbit.radius_km
    eta_limb = box.el_nadir_deg - box.el_limb_deg
    v_g = orbit.ground_speed_km_s(lat_abs)
    slant_limb = slant_at_eta_km(eta_limb, r_iss, r_earth)
    swath_fov = cross_track_width_km(optics.half_az_deg, slant_limb)
    swath_box = cross_track_width_km(box.az_box_deg, slant_limb)

    omega_el_limb = scene_el_rate_deg_s(orbit, box, lat_abs, eta_limb)
    omega_az = az_scan_rate_deg_s(omega_rel_deg_s, omega_el_limb)
    if omega_az <= 1e-12:
        fill = 0.0
    else:
        t_round_s = 4.0 * box.az_box_deg / omega_az
        incidence = math.degrees(
            math.asin(min(1.0, (r_iss / r_earth) * math.sin(math.radians(eta_limb))))
        )
        along_fp = cross_track_width_km(optics.half_el_deg, slant_limb)
        along_fp = along_fp / max(1e-6, math.cos(math.radians(incidence)))
        gap = v_g * t_round_s
        fill = 1.0 if gap <= 1e-9 else min(1.0, along_fp / gap)

    swath_limb_1 = swath_fov
    swath_limb_2 = swath_fov + fill * (swath_box - swath_fov)

    def _rewind_leg(t_dwell_s: float) -> tuple[float, float, float]:
        el_loss = _el_loss_deg(box, t_dwell_s, t_window_s)
        t_rewind = _rewind_time_s(el_loss, box, omega_img_deg_s)
        eta_loss = box.el_nadir_deg - el_loss
        eta_mid = 0.5 * (eta_loss + eta_limb)
        ds_deta = d_look_ahead_d_eta_km_per_rad(eta_mid, r_iss, r_earth)
        v_rewind = v_g + ds_deta * math.radians(omega_img_deg_s)
        slant_mid = slant_at_eta_km(eta_mid, r_iss, r_earth)
        swath_rewind = cross_track_width_km(optics.half_az_deg, slant_mid)
        return t_rewind, v_rewind, swath_rewind

    t_rw1, v_rw1, w_rw1 = _rewind_leg(t_dwell_1_s)
    t_rw2, v_rw2, w_rw2 = _rewind_leg(t_dwell_2_s)

    lam_a1 = dens_per_km2 * w_rw1 * v_rw1
    lam_a2 = dens_per_km2 * w_rw2 * v_rw2
    lam_b1 = dens_per_km2 * swath_limb_1 * v_g
    lam_b2 = dens_per_km2 * swath_limb_2 * v_g
    return HuntResult(
        t_reacq_1_s=two_phase_mean_wait_s(lam_a1, t_rw1, lam_b1),
        t_reacq_2_s=two_phase_mean_wait_s(lam_a2, t_rw2, lam_b2),
        t_rewind_1_s=t_rw1,
        t_rewind_2_s=t_rw2,
        omega_az_scan_deg_s=omega_az,
        scan_fill=fill,
        swath_fov_km=swath_fov,
        swath_box_km=swath_box,
    )
