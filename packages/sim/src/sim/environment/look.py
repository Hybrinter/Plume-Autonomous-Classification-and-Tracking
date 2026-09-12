"""Look angles from ISS ECI to an ECEF CoG in the flight mount convention."""

from __future__ import annotations

import math

# third-party
import numpy as np

# internal
from flight.hal.interfaces.ephemeris import IssState
from flight.payload.gimbal.geo import ecef_from_eci, eci_from_ecef, lvlh_axes

from sim.environment.models.earth import EarthModel
from sim.environment.records import LookAngles

_OCCLUDE_EPS_M = 1.0


def earth_occludes_cog(
    iss: IssState,
    cog_ecef_m: tuple[float, float, float],
    earth: EarthModel,
    omega_earth_rad_s: float,
    epoch_utc_s: float,
) -> bool:
    """Return True when the nearest height-0 Earth hit is closer than the CoG.

    Args:
        iss: Truth ISS ECI state.
        cog_ecef_m: Plume CoG in ECEF meters.
        earth: Earth model (height-proxy intersect).
        omega_earth_rad_s: Earth rotation rate.
        epoch_utc_s: ECI/ECEF alignment epoch.

    Returns:
        True if an Earth surface hit lies in front of the CoG. A ray that
        misses Earth is not occlusion.
    """
    r_iss = np.asarray(iss.r_m, dtype=np.float64)
    cog_ecef = np.asarray(cog_ecef_m, dtype=np.float64)
    cog_eci = eci_from_ecef(cog_ecef, omega_earth_rad_s, iss.epoch_utc_s, epoch_utc_s)
    look = cog_eci - r_iss
    slant = float(np.linalg.norm(look))
    if slant < 1.0:
        return False
    unit = look / slant
    r_iss_ecef = ecef_from_eci(r_iss, omega_earth_rad_s, iss.epoch_utc_s, epoch_utc_s)
    d_ecef = ecef_from_eci(unit, omega_earth_rad_s, iss.epoch_utc_s, epoch_utc_s)
    d_n = float(np.linalg.norm(d_ecef))
    if d_n <= 1e-18:
        return False
    hit = earth.intersect_at_height(r_iss_ecef, d_ecef / d_n, 0.0)
    if hit is None:
        return False
    _point, hit_slant = hit
    return float(hit_slant) + _OCCLUDE_EPS_M < slant


def look_angles_at(
    iss: IssState,
    cog_ecef_m: tuple[float, float, float],
    earth: EarthModel,
    omega_earth_rad_s: float,
    epoch_utc_s: float,
) -> LookAngles:
    """Return mount look angles. Elevation 0 at nadir, positive along-track.

    Args:
        iss: Truth ISS ECI state.
        cog_ecef_m: Plume CoG in ECEF meters.
        earth: Earth model (visibility via height-proxy intersect).
        omega_earth_rad_s: Earth rotation rate.
        epoch_utc_s: ECI/ECEF alignment epoch.

    Returns:
        LookAngles. visible is an unoccluded height-0 Earth hit, not FOV.
    """
    r_iss = np.asarray(iss.r_m, dtype=np.float64)
    v_iss = np.asarray(iss.v_m_s, dtype=np.float64)
    cog_ecef = np.asarray(cog_ecef_m, dtype=np.float64)
    cog_eci = eci_from_ecef(cog_ecef, omega_earth_rad_s, iss.epoch_utc_s, epoch_utc_s)
    look = cog_eci - r_iss
    slant = float(np.linalg.norm(look))
    if slant < 1.0:
        return LookAngles(0.0, 0.0, 0.0, 0.0, 0.0, False)
    unit = look / slant
    x_hat, y_hat, z_hat = lvlh_axes(r_iss, v_iss)
    lx = float(np.dot(look, x_hat))
    ly = float(np.dot(look, y_hat))
    lz = float(np.dot(look, z_hat))
    el_rad = math.atan2(lx, lz)
    az_rad = math.atan2(ly, math.hypot(lx, lz))
    eta_rad = math.acos(float(np.clip(np.dot(unit, z_hat), -1.0, 1.0)))
    r_sensor = float(np.linalg.norm(r_iss))
    r_target = float(np.linalg.norm(cog_eci))
    if r_target <= 1e-9:
        inc_rad = 0.0
    else:
        sine_i = (r_sensor / r_target) * math.sin(eta_rad)
        sine_i = min(1.0, max(0.0, sine_i))
        inc_rad = math.asin(sine_i)
    r_iss_ecef = ecef_from_eci(r_iss, omega_earth_rad_s, iss.epoch_utc_s, epoch_utc_s)
    d_ecef = ecef_from_eci(unit, omega_earth_rad_s, iss.epoch_utc_s, epoch_utc_s)
    d_n = float(np.linalg.norm(d_ecef))
    visible = False
    if d_n > 1e-18:
        hit = earth.intersect_at_height(r_iss_ecef, d_ecef / d_n, 0.0)
        if hit is not None:
            _point, hit_slant = hit
            visible = float(hit_slant) + _OCCLUDE_EPS_M >= slant
    return LookAngles(
        az_rad=az_rad,
        el_rad=el_rad,
        eta_rad=eta_rad,
        slant_m=slant,
        incidence_rad=inc_rad,
        visible=visible,
    )
