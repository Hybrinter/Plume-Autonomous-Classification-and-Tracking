"""Pinhole CoG / boresight ray and height-ellipsoid Earth intersect (pure).

Each accepted vision frame rebuilds the line of sight through the blob center of
geometry, rotates it into ECI, and intersects a constant-height WGS-84 ellipsoid.
The hit is stored in ECEF meters. A miss keeps the last good CoG.
`intersect_boresight` uses the principal-point ray at the current elevation.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# third-party
import numpy as np

# internal
from flight.payload.gimbal.geo import (
    boresight_mount,
    cam_ray_to_mount,
    ecef_from_eci,
    mount_to_eci,
    pinhole_cam_ray,
    wgs84_intersect_at_height,
)


@dataclass(frozen=True, slots=True)
class IntersectResult:
    """Earth intersect of one CoG or boresight ray.

    Attributes:
        r_cog_ecef_m: Hit in ECEF meters, or None on a miss.
        hit: True when the ray meets the height ellipsoid in front of the camera.
        slant_m: Intermediate slant range; discarded by the caller after this return.
    """

    r_cog_ecef_m: tuple[float, float, float] | None
    hit: bool
    slant_m: float


def _hit_from_eci_ray(
    r_iss_eci_m: np.ndarray,
    d_eci: np.ndarray,
    utc_s: float,
    epoch_utc_s: float,
    omega_earth_rad_s: float,
    wgs84_a_m: float,
    wgs84_f: float,
    last_r_cog_ecef_m: tuple[float, float, float] | None,
    height_m: float,
) -> IntersectResult:
    """Intersect one ECI look direction with the height ellipsoid in ECEF.

    Inputs:
        r_iss_eci_m: ISS ECI position meters, shape (3,).
        d_eci: Unit ECI look direction, shape (3,).
        utc_s, epoch_utc_s: UTC seconds for ECI/ECEF rotation.
        omega_earth_rad_s: Earth rotation rate.
        wgs84_a_m, wgs84_f: Surface ellipsoid scalars.
        last_r_cog_ecef_m: Previous hit, kept on a miss.
        height_m: Geodetic-height offset meters (tracking proxy).

    Outputs:
        IntersectResult: New ECEF point on a hit; last CoG and hit=False on a miss.
    """
    r_iss_ecef = ecef_from_eci(r_iss_eci_m, omega_earth_rad_s, utc_s, epoch_utc_s)
    d_ecef = ecef_from_eci(d_eci, omega_earth_rad_s, utc_s, epoch_utc_s)
    d_ecef_n = float(np.linalg.norm(d_ecef))
    if d_ecef_n < 1e-18:
        return IntersectResult(r_cog_ecef_m=last_r_cog_ecef_m, hit=False, slant_m=0.0)
    d_ecef = d_ecef / d_ecef_n  # np.ndarray[float64, (3,)]
    hit = wgs84_intersect_at_height(r_iss_ecef, d_ecef, wgs84_a_m, wgs84_f, height_m)
    if hit is None:
        return IntersectResult(r_cog_ecef_m=last_r_cog_ecef_m, hit=False, slant_m=0.0)
    point, slant = hit
    return IntersectResult(
        r_cog_ecef_m=(float(point[0]), float(point[1]), float(point[2])),
        hit=True,
        slant_m=float(slant),
    )


def intersect_cog(
    p_cog_px: tuple[float, float],
    theta_g_rad: float,
    r_iss_eci_m: tuple[float, float, float],
    v_iss_eci_m_s: tuple[float, float, float],
    utc_s: float,
    epoch_utc_s: float,
    omega_earth_rad_s: float,
    wgs84_a_m: float,
    wgs84_f: float,
    plane_width_px: int,
    plane_height_px: int,
    pixel_pitch_m: float,
    focal_m: float,
    last_r_cog_ecef_m: tuple[float, float, float] | None,
    height_m: float,
) -> IntersectResult:
    """Intersect the CoG pinhole ray with the constant-height ellipsoid.

    Inputs:
        p_cog_px: Band-plane centroid (u, v).
        theta_g_rad: Gimbal elevation at shutter.
        r_iss_eci_m, v_iss_eci_m_s: ISS ECI state.
        utc_s, epoch_utc_s: UTC seconds for ECI/ECEF rotation.
        omega_earth_rad_s: Earth rotation rate.
        wgs84_a_m, wgs84_f: Surface ellipsoid scalars.
        plane_width_px, plane_height_px: Band-plane size.
        pixel_pitch_m, focal_m: Pinhole geometry (band pitch, focal length).
        last_r_cog_ecef_m: Previous hit, kept on a miss.
        height_m: Geodetic-height offset meters (tracking proxy).

    Outputs:
        IntersectResult: New ECEF CoG on a hit; last CoG and hit=False on a miss.
    """
    r_iss = np.asarray(r_iss_eci_m, dtype=np.float64)  # np.ndarray[float64, (3,)]
    v_iss = np.asarray(v_iss_eci_m_s, dtype=np.float64)  # np.ndarray[float64, (3,)]
    d_cam = pinhole_cam_ray(
        p_cog_px[0],
        p_cog_px[1],
        plane_width_px,
        plane_height_px,
        pixel_pitch_m,
        focal_m,
    )
    d_mount = cam_ray_to_mount(d_cam, theta_g_rad)
    d_eci = mount_to_eci(d_mount, r_iss, v_iss)
    return _hit_from_eci_ray(
        r_iss,
        d_eci,
        utc_s,
        epoch_utc_s,
        omega_earth_rad_s,
        wgs84_a_m,
        wgs84_f,
        last_r_cog_ecef_m,
        height_m,
    )


def intersect_boresight(
    theta_g_rad: float,
    r_iss_eci_m: tuple[float, float, float],
    v_iss_eci_m_s: tuple[float, float, float],
    utc_s: float,
    epoch_utc_s: float,
    omega_earth_rad_s: float,
    wgs84_a_m: float,
    wgs84_f: float,
    last_r_cog_ecef_m: tuple[float, float, float] | None,
    height_m: float,
) -> IntersectResult:
    """Intersect the principal-point ray with the constant-height ellipsoid.

    Inputs:
        theta_g_rad: Gimbal elevation at shutter.
        r_iss_eci_m, v_iss_eci_m_s: ISS ECI state.
        utc_s, epoch_utc_s: UTC seconds for ECI/ECEF rotation.
        omega_earth_rad_s: Earth rotation rate.
        wgs84_a_m, wgs84_f: Surface ellipsoid scalars.
        last_r_cog_ecef_m: Previous hit, kept on a miss.
        height_m: Geodetic-height offset meters (tracking proxy).

    Outputs:
        IntersectResult: New ECEF point on a hit; last CoG and hit=False on a miss.

    Notes:
        The ray is mount boresight at theta_g_rad (camera principal point). Callers
        use this for REWIND / no-plume scene rate. A miss keeps last_r_cog_ecef_m.
    """
    r_iss = np.asarray(r_iss_eci_m, dtype=np.float64)  # np.ndarray[float64, (3,)]
    v_iss = np.asarray(v_iss_eci_m_s, dtype=np.float64)  # np.ndarray[float64, (3,)]
    d_mount = boresight_mount(theta_g_rad)
    d_eci = mount_to_eci(d_mount, r_iss, v_iss)
    return _hit_from_eci_ray(
        r_iss,
        d_eci,
        utc_s,
        epoch_utc_s,
        omega_earth_rad_s,
        wgs84_a_m,
        wgs84_f,
        last_r_cog_ecef_m,
        height_m,
    )
