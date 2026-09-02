"""Gimbal pointing box and look angles in the payload body frame.

Elevation is 90 deg at geocentric nadir and decreases toward the limb along
the velocity vector. Azimuth is positive toward the right (cross-track).
The body +Z axis is nadir, +X is along-track (horizontal), +Y completes
the right-handed frame.

Contains:
  - WindowMode / GimbalBox: operational elevation window and azimuth stop.
  - Look: azimuth, elevation, off-nadir, slant, Earth-hit flag.
  - rotate_z / body_axes / look_at / ray_hits_earth / earth_hit.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np


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
        el_limb_deg: Elevation at the limb stop (30 for the polar-slice view).
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
        visible: True when the line of sight intersects the Earth sphere.
    """

    az_deg: float
    el_deg: float
    eta_deg: float
    slant_km: float
    visible: bool


def rotate_z(theta_rad: float) -> np.ndarray:
    """Return a right-handed rotation matrix about +Z.

    Args:
        theta_rad: Rotation angle in radians.

    Returns:
        3x3 rotation matrix. # np.ndarray[float64, (3, 3)]
    """
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])


def body_axes(r_iss: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return payload body axes: +X along-track, +Y right, +Z nadir.

    Args:
        r_iss: ISS ECI position in kilometres. # np.ndarray[float64, (3,)]
        vel: ISS inertial velocity in km/s. # np.ndarray[float64, (3,)]

    Returns:
        Unit axes (x, y, z).
    """
    z_axis = -r_iss / np.linalg.norm(r_iss)
    x_axis = vel - np.dot(vel, z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return x_axis, y_axis, z_axis


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
    x_axis, y_axis, z_axis = body_axes(r_iss, vel)
    look = target_eci - r_iss
    slant = float(np.linalg.norm(look))
    if slant < 1e-9:
        return Look(0.0, el_nadir_deg, 0.0, 0.0, False)
    unit = look / slant
    lx = float(np.dot(look, x_axis))
    ly = float(np.dot(look, y_axis))
    lz = float(np.dot(look, z_axis))
    eta = math.degrees(math.acos(float(np.clip(np.dot(unit, z_axis), -1.0, 1.0))))
    along_off = math.degrees(math.atan2(lx, lz))
    el_deg = el_nadir_deg - along_off
    az_deg = math.degrees(math.atan2(ly, math.hypot(lx, lz)))
    vis = ray_hits_earth(r_iss, unit, float(np.linalg.norm(target_eci)))
    return Look(az_deg=az_deg, el_deg=el_deg, eta_deg=eta, slant_km=slant, visible=vis)


def ray_hits_earth(sensor: np.ndarray, unit: np.ndarray, earth_r_km: float) -> bool:
    """Return True if the ray ``sensor + t unit`` hits the sphere of radius ``earth_r_km``.

    Args:
        sensor: Ray origin in kilometres. # np.ndarray[float64, (3,)]
        unit: Unit direction. # np.ndarray[float64, (3,)]
        earth_r_km: Sphere radius in kilometres.

    Returns:
        True when at least one positive-t intersection exists.
    """
    b_coef = 2.0 * float(np.dot(sensor, unit))
    c_coef = float(np.dot(sensor, sensor)) - earth_r_km**2
    disc = b_coef * b_coef - 4.0 * c_coef
    if disc < 0.0:
        return False
    sqrt_d = math.sqrt(disc)
    t1 = (-b_coef - sqrt_d) / 2.0
    t2 = (-b_coef + sqrt_d) / 2.0
    return t1 > 0.0 or t2 > 0.0


def earth_hit(sensor: np.ndarray, unit: np.ndarray, earth_r_km: float) -> np.ndarray | None:
    """Return the nearest forward intersection with the Earth sphere, or None.

    Args:
        sensor: Ray origin in kilometres. # np.ndarray[float64, (3,)]
        unit: Unit direction. # np.ndarray[float64, (3,)]
        earth_r_km: Sphere radius in kilometres.

    Returns:
        Intersection point in kilometres, or None. # np.ndarray[float64, (3,)]
    """
    b_coef = 2.0 * float(np.dot(sensor, unit))
    c_coef = float(np.dot(sensor, sensor)) - earth_r_km**2
    disc = b_coef * b_coef - 4.0 * c_coef
    if disc < 0.0:
        return None
    sqrt_d = math.sqrt(disc)
    times = [t for t in ((-b_coef - sqrt_d) / 2.0, (-b_coef + sqrt_d) / 2.0) if t > 1e-6]
    if not times:
        return None
    return sensor + min(times) * unit
