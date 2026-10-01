"""Pure per-pixel and per-tile ground sample distance on the Earth ellipsoid.

H is along-track, W is lateral. Image down points look-back, so positive
elevation produces smaller footprints toward the bottom of the frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from flight.libs.types import Err, FaultCode, Ok, Result
from flight.payload.gimbal.intersect import CameraGeometry, intersect_cog

GSD_REFERENCE_M = 15.87
Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class GsdPair:
    """Ground metres per pixel, ordered lateral (W), then along-track (H)."""

    lateral_m: float
    along_m: float


def pixel_gsd(
    u: float,
    v: float,
    theta_g_rad: float,
    r_iss_eci_m: Vec3,
    v_iss_eci_m_s: Vec3,
    utc_s: float,
    epoch_utc_s: float,
    omega_earth_rad_s: float,
    wgs84_a_m: float,
    wgs84_f: float,
    camera: CameraGeometry,
    height_m: float = 0.0,
) -> GsdPair | None:
    """Intersect symmetric half-pixel rays and measure their ECEF separation.

    Return None for invalid geometry, degenerate orbit axes, or a missed ray.
    Half-pixel differences center both footprint measurements on (u, v).
    Chord lengths approximate the local ground distances at pixel scale.
    """
    scalars = (
        u,
        v,
        theta_g_rad,
        utc_s,
        epoch_utc_s,
        omega_earth_rad_s,
        wgs84_a_m,
        wgs84_f,
        camera.pixel_pitch_m,
        camera.focal_length_m,
        height_m,
        *r_iss_eci_m,
        *v_iss_eci_m_s,
    )
    if not all(math.isfinite(value) for value in scalars):
        return None
    if (
        camera.width_px < 1
        or camera.height_px < 1
        or camera.pixel_pitch_m <= 0
        or camera.focal_length_m <= 0
        or wgs84_a_m <= 0
        or not 0 <= wgs84_f < 1
        or height_m < 0
        or np.linalg.norm(r_iss_eci_m) <= wgs84_a_m
        or np.linalg.norm(np.cross(r_iss_eci_m, v_iss_eci_m_s)) < 1e-12
    ):
        return None
    points: list[np.ndarray] = []
    for point in ((u - 0.5, v), (u + 0.5, v), (u, v - 0.5), (u, v + 0.5)):
        hit = intersect_cog(
            point,
            theta_g_rad,
            r_iss_eci_m,
            v_iss_eci_m_s,
            utc_s,
            epoch_utc_s,
            omega_earth_rad_s,
            wgs84_a_m,
            wgs84_f,
            camera,
            height_m,
        )
        if hit is None:
            return None
        points.append(np.asarray(hit.point_ecef_m, dtype=np.float64))
    lateral = float(np.linalg.norm(points[1] - points[0]))
    along = float(np.linalg.norm(points[3] - points[2]))
    if not all(math.isfinite(value) and value > 0 for value in (lateral, along)):
        return None
    return GsdPair(lateral, along)


def tile_gsd_grid(
    theta_g_rad: float,
    r_iss_eci_m: Vec3,
    v_iss_eci_m_s: Vec3,
    utc_s: float,
    epoch_utc_s: float,
    omega_earth_rad_s: float,
    wgs84_a_m: float,
    wgs84_f: float,
    camera: CameraGeometry,
    grid: tuple[int, int] = (8, 8),
    height_m: float = 0.0,
) -> np.ndarray | None:
    """Return (rows, cols, 2) float32 metres at tile centers, or None.

    Pixel coordinates use the same frame-edge convention as intersect_cog.
    A tile spans [col*width, (col+1)*width] with center at the midpoint.
    """
    rows, cols = grid
    if rows < 1 or cols < 1 or camera.height_px % rows or camera.width_px % cols:
        return None
    height, width = camera.height_px // rows, camera.width_px // cols
    values = np.empty((rows, cols, 2), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            pair = pixel_gsd(
                (col + 0.5) * width,
                (row + 0.5) * height,
                theta_g_rad,
                r_iss_eci_m,
                v_iss_eci_m_s,
                utc_s,
                epoch_utc_s,
                omega_earth_rad_s,
                wgs84_a_m,
                wgs84_f,
                camera,
                height_m,
            )
            if pair is None:
                return None
            values[row, col] = (pair.lateral_m, pair.along_m)
    return values


def nominal_iss_state(
    altitude_m: float,
    wgs84_a_m: float = 6_378_137.0,
    mu_m3_s2: float = 3.986004418e14,
) -> tuple[Vec3, Vec3, float] | None:
    """Return an equatorial circular ECI orbit at epoch zero, or None.

    This reference orbit is for source adaptation and explicitly flagged
    degraded inference, never a replacement for measured ephemeris.
    """
    if not all(math.isfinite(value) and value > 0 for value in (altitude_m, wgs84_a_m, mu_m3_s2)):
        return None
    radius = wgs84_a_m + altitude_m
    return (radius, 0.0, 0.0), (0.0, math.sqrt(mu_m3_s2 / radius), 0.0), 0.0


def to_model_gsd(
    gsd_m: np.ndarray,
    reference_m: float = GSD_REFERENCE_M,
) -> Result[np.ndarray, FaultCode]:
    """Encode (..., 2) metres as float32 ln(GSD/reference), or malformed input."""
    values = np.asarray(gsd_m, dtype=np.float64)
    if (
        not math.isfinite(reference_m)
        or reference_m <= 0
        or values.ndim < 1
        or values.shape[-1] != 2
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        return Err(FaultCode.FRAME_MALFORMED)
    return Ok(np.log(values / reference_m).astype(np.float32))
