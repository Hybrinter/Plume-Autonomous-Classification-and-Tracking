"""Gimbal pointing box and look angles in the payload body frame.

km/deg facade over flight LVLH axes and ``sim.environment.look``. Elevation is
90 deg at geocentric nadir and decreases toward the limb along the velocity
vector. Azimuth is positive toward the right (cross-track). The body +Z axis
is nadir, +X is along-track (horizontal), +Y completes the right-handed frame.

Contains:
  - WindowMode / GimbalBox: operational elevation window and azimuth stop.
  - Look: azimuth, elevation, off-nadir, slant, incidence, Earth-hit flag.
  - look_from_si: flight-convention SI look to this km/deg Look.
  - rotate_z / body_axes / look_at / incidence_deg / ray_hits_earth / earth_hit.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
from flight.hal.interfaces.ephemeris import IssState
from flight.payload.gimbal.geo import lvlh_axes, rz
from sim.environment.look import look_angles_at
from sim.environment.models.earth import SphereEarth
from sim.environment.records import LookAngles


class WindowMode(enum.StrEnum):
    """Elevation-window shape.

    String values mirror member names.
    """

    ONE_SIDED = "ONE_SIDED"
    TWO_SIDED = "TWO_SIDED"


@dataclass(frozen=True)
class GimbalBox:
    """Hard gimbal stops and elevation-window shape.

    Attributes:
        az_box_deg: Azimuth stop from boresight, degrees (two-axis box).
        el_nadir_deg: Elevation at geocentric nadir (90 for a nadir mount).
        el_limb_deg: Elevation at the science limb stop (90 minus eta_max).
        window_mode: One-sided (nadir to limb) or symmetric about nadir.
    """

    az_box_deg: float
    el_nadir_deg: float
    el_limb_deg: float
    window_mode: WindowMode


@dataclass(frozen=True)
class Look:
    """Look angles from the payload to a target.

    Attributes:
        az_deg: Azimuth in degrees (positive right / cross-track).
        el_deg: Elevation in degrees (90 at nadir).
        eta_deg: Off-nadir angle in degrees.
        slant_km: Slant range in kilometres.
        incidence_deg: Earth emission angle at the target, degrees.
        visible: True when the line of sight intersects the Earth sphere.
    """

    az_deg: float
    el_deg: float
    eta_deg: float
    slant_km: float
    incidence_deg: float
    visible: bool


def look_from_si(si: LookAngles, el_nadir_deg: float) -> Look:
    """Map flight-convention SI look angles onto the km/deg analysis Look.

    Flight elevation is 0 rad at nadir. This Look stores elevation as
    ``el_nadir_deg`` minus that off-nadir angle in degrees.

    Args:
        si: SI look from ``sim.environment.look.look_angles_at``.
        el_nadir_deg: Elevation assigned to geocentric nadir (usually 90).

    Returns:
        Look in kilometres and degrees.
    """
    return Look(
        az_deg=math.degrees(si.az_rad),
        el_deg=el_nadir_deg - math.degrees(si.el_rad),
        eta_deg=math.degrees(si.eta_rad),
        slant_km=si.slant_m / 1000.0,
        incidence_deg=math.degrees(si.incidence_rad),
        visible=si.visible,
    )


def rotate_z(theta_rad: float) -> np.ndarray:
    """Return a right-handed rotation matrix about +Z.

    Args:
        theta_rad: Rotation angle in radians.

    Returns:
        3x3 rotation matrix. # np.ndarray[float64, (3, 3)]
    """
    return rz(theta_rad)


def body_axes(r_iss: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return payload body axes: +X along-track, +Y right, +Z nadir.

    Args:
        r_iss: ISS ECI position in kilometres. # np.ndarray[float64, (3,)]
        vel: ISS inertial velocity in km/s. # np.ndarray[float64, (3,)]

    Returns:
        Unit axes (x, y, z).
    """
    return lvlh_axes(np.asarray(r_iss, dtype=np.float64), np.asarray(vel, dtype=np.float64))


def look_at(
    r_iss: np.ndarray,
    vel: np.ndarray,
    target_eci: np.ndarray,
    el_nadir_deg: float,
) -> Look:
    """Return look angles from the payload to ``target_eci``.

    Args:
        r_iss: ISS ECI position in kilometres. # np.ndarray[float64, (3,)]
        vel: ISS inertial velocity in km/s. # np.ndarray[float64, (3,)]
        target_eci: Target ECI position in kilometres. # np.ndarray[float64, (3,)]
        el_nadir_deg: Elevation assigned to geocentric nadir (usually 90).

    Returns:
        Look angles. ``visible`` is the Earth-sphere hit test, not FOV.
    """
    r = np.asarray(r_iss, dtype=np.float64)
    vel_vec = np.asarray(vel, dtype=np.float64)
    tgt = np.asarray(target_eci, dtype=np.float64)
    look_vec = tgt - r
    slant = float(np.linalg.norm(look_vec))
    if slant < 1e-9:
        return Look(0.0, el_nadir_deg, 0.0, 0.0, 0.0, False)
    iss = IssState(
        r_m=(float(r[0]) * 1000.0, float(r[1]) * 1000.0, float(r[2]) * 1000.0),
        v_m_s=(float(vel_vec[0]) * 1000.0, float(vel_vec[1]) * 1000.0, float(vel_vec[2]) * 1000.0),
        epoch_utc_s=0.0,
    )
    cog_m = (float(tgt[0]) * 1000.0, float(tgt[1]) * 1000.0, float(tgt[2]) * 1000.0)
    earth = SphereEarth(a_m=float(np.linalg.norm(tgt)) * 1000.0)
    si = look_angles_at(iss, cog_m, earth, 0.0, 0.0)
    return look_from_si(si, el_nadir_deg)


def incidence_deg(eta_deg: float, r_sensor_km: float, r_target_km: float) -> float:
    """Return Earth incidence (emission) angle from off-nadir look angle.

    Uses the spherical law of sines: sin(i) / r_sensor = sin(eta) / r_target.

    Args:
        eta_deg: Off-nadir angle in degrees.
        r_sensor_km: Sensor geocentric radius in kilometres.
        r_target_km: Target geocentric radius in kilometres.

    Returns:
        Incidence angle in degrees, 0 at nadir, 90 at the geometric limb.
    """
    if r_target_km <= 1e-9:
        return 0.0
    sine_i = (r_sensor_km / r_target_km) * math.sin(math.radians(eta_deg))
    sine_i = min(1.0, max(0.0, sine_i))
    return math.degrees(math.asin(sine_i))


def ray_hits_earth(sensor: np.ndarray, unit: np.ndarray, earth_r_km: float) -> bool:
    """Return True if the ray ``sensor + t unit`` hits the sphere of radius ``earth_r_km``.

    Args:
        sensor: Ray origin in kilometres. # np.ndarray[float64, (3,)]
        unit: Unit direction. # np.ndarray[float64, (3,)]
        earth_r_km: Sphere radius in kilometres.

    Returns:
        True when at least one positive-t intersection exists.
    """
    earth = SphereEarth(a_m=earth_r_km * 1000.0)
    origin = np.asarray(sensor, dtype=np.float64) * 1000.0
    direction = np.asarray(unit, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-18:
        return False
    return earth.intersect_at_height(origin, direction / norm, 0.0) is not None


def earth_hit(sensor: np.ndarray, unit: np.ndarray, earth_r_km: float) -> np.ndarray | None:
    """Return the nearest forward intersection with the Earth sphere, or None.

    Args:
        sensor: Ray origin in kilometres. # np.ndarray[float64, (3,)]
        unit: Unit direction. # np.ndarray[float64, (3,)]
        earth_r_km: Sphere radius in kilometres.

    Returns:
        Intersection point in kilometres, or None. # np.ndarray[float64, (3,)]
    """
    earth = SphereEarth(a_m=earth_r_km * 1000.0)
    origin = np.asarray(sensor, dtype=np.float64) * 1000.0
    direction = np.asarray(unit, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-18:
        return None
    hit = earth.intersect_at_height(origin, direction / norm, 0.0)
    if hit is None:
        return None
    point, _slant = hit
    return np.asarray(point, dtype=np.float64) / 1000.0
