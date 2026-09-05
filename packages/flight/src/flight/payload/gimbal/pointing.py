"""Pinhole boresight-relative pointing geometry (pure).

Elevation error uses the pinhole (not px * IFOV). Image +x is unactuated optical
azimuth. Image +y (down) is -elevation. Band-plane pitch is 2 * mosaic pixel pitch.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
import math

# internal
from flight.payload.gimbal.geo import pinhole_cam_ray


def pinhole_error_rad(
    centroid_px: tuple[float, float],
    plane_width_px: int,
    plane_height_px: int,
    pixel_pitch_m: float,
    focal_m: float,
) -> tuple[float, float]:
    """Optical (az, el) boresight error of a centroid, radians.

    Inputs:
        centroid_px: (u, v) in band-plane pixels.
        plane_width_px, plane_height_px: Band-plane size; principal point at center.
        pixel_pitch_m: Band-plane pitch in meters (2 * mosaic pixel pitch).
        focal_m: Focal length in meters.

    Outputs:
        tuple[float, float]: (e_az_rad, e_el_rad). e_az is unactuated. Image +y is -e.
    """
    d_cam = pinhole_cam_ray(
        centroid_px[0],
        centroid_px[1],
        plane_width_px,
        plane_height_px,
        pixel_pitch_m,
        focal_m,
    )
    e_az = math.atan2(float(d_cam[0]), float(d_cam[2]))
    e_el = -math.atan2(float(d_cam[1]), float(d_cam[2]))
    return (e_az, e_el)


def boresight_error_deg(
    centroid_px: tuple[float, float],
    plane_width_px: int,
    plane_height_px: int,
    pixel_pitch_m: float,
    focal_m: float,
) -> tuple[float, float]:
    """Optical (az, el) boresight error of a centroid, degrees.

    Inputs:
        centroid_px: (u, v) in band-plane pixels.
        plane_width_px, plane_height_px: Band-plane size.
        pixel_pitch_m: Band-plane pitch in meters.
        focal_m: Focal length in meters.

    Outputs:
        tuple[float, float]: (e_az_deg, e_el_deg).
    """
    e_az, e_el = pinhole_error_rad(
        centroid_px, plane_width_px, plane_height_px, pixel_pitch_m, focal_m
    )
    return (math.degrees(e_az), math.degrees(e_el))


def target_displacement_px(
    centroid_px: tuple[float, float],
    plane_width_px: int,
    plane_height_px: int,
) -> float:
    """Euclidean band-plane pixel distance of the centroid from boresight.

    Inputs:
        centroid_px: (x, y) centroid in band-plane pixel coordinates.
        plane_width_px: Width of the band plane in pixels.
        plane_height_px: Height of the band plane in pixels.

    Outputs:
        float: Euclidean distance in band-plane pixels from boresight to the centroid.
    """
    return math.hypot(centroid_px[0] - plane_width_px / 2.0, centroid_px[1] - plane_height_px / 2.0)
