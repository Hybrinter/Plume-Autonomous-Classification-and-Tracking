"""Pinhole CoG ray and WGS-84 Earth intersect (pure).

Each accepted vision frame rebuilds the line of sight through the blob center of
geometry, rotates it into ECI, and intersects the WGS-84 ellipsoid. The hit is
stored in ECEF meters. A miss keeps the last good CoG.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# third-party
import numpy as np

# internal
from flight.payload.gimbal.geo import (
    cam_ray_to_mount,
    ecef_from_eci,
    mount_to_eci,
    pinhole_cam_ray,
    wgs84_intersect,
)


@dataclass(frozen=True, slots=True)
class IntersectResult:
    """Earth intersect of one CoG ray.

    Attributes:
        r_cog_ecef_m: Hit in ECEF meters, or None on a miss.
        hit: True when the ray meets the ellipsoid in front of the camera.
        slant_m: Intermediate slant range; discarded by the caller after this return.
    """

    r_cog_ecef_m: tuple[float, float, float] | None
    hit: bool
    slant_m: float


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
) -> IntersectResult:
    """Intersect the CoG pinhole ray with WGS-84.

    Inputs:
        p_cog_px: Band-plane centroid (u, v).
        theta_g_rad: Gimbal elevation at shutter.
        r_iss_eci_m, v_iss_eci_m_s: ISS ECI state.
        utc_s, epoch_utc_s: UTC seconds for ECI/ECEF rotation.
        omega_earth_rad_s: Earth rotation rate.
        wgs84_a_m, wgs84_f: Ellipsoid scalars.
        plane_width_px, plane_height_px: Band-plane size.
        pixel_pitch_m, focal_m: Pinhole geometry (band pitch, focal length).
        last_r_cog_ecef_m: Previous hit, kept on a miss.

    Outputs:
        IntersectResult: New ECEF CoG on a hit; last CoG and hit=False on a miss.
    """
    r_iss = np.asarray(r_iss_eci_m, dtype=np.float64)
    v_iss = np.asarray(v_iss_eci_m_s, dtype=np.float64)
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
    r_iss_ecef = ecef_from_eci(r_iss, omega_earth_rad_s, utc_s, epoch_utc_s)
    d_ecef = ecef_from_eci(d_eci, omega_earth_rad_s, utc_s, epoch_utc_s)
    d_ecef_n = float(np.linalg.norm(d_ecef))
    if d_ecef_n < 1e-18:
        return IntersectResult(r_cog_ecef_m=last_r_cog_ecef_m, hit=False, slant_m=0.0)
    d_ecef = d_ecef / d_ecef_n
    hit = wgs84_intersect(r_iss_ecef, d_ecef, wgs84_a_m, wgs84_f)
    if hit is None:
        return IntersectResult(r_cog_ecef_m=last_r_cog_ecef_m, hit=False, slant_m=0.0)
    point, slant = hit
    return IntersectResult(
        r_cog_ecef_m=(float(point[0]), float(point[1]), float(point[2])),
        hit=True,
        slant_m=float(slant),
    )
