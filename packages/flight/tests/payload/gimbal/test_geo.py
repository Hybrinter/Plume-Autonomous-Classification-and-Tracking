"""Tests for mount / LVLH / WGS-84 geometry helpers."""

import math

import numpy as np
from flight.payload.gimbal.geo import (
    boresight_mount,
    pinhole_cam_ray,
    rx_neg,
    wgs84_intersect,
)


def test_boresight_matches_rx_neg_nadir() -> None:
    """Boresight (0, sin theta, cos theta) is R_x(-theta) applied to nadir +z."""
    theta = math.radians(20.0)
    b = boresight_mount(theta)
    mapped = rx_neg(theta) @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(b, mapped)
    assert abs(b[0]) < 1e-15


def test_centered_pinhole_is_camera_plus_z() -> None:
    """The principal-point ray is camera +Z."""
    ray = pinhole_cam_ray(612.0, 512.0, 1224, 1024, 6.9e-6, 0.150)
    assert np.allclose(ray, np.array([0.0, 0.0, 1.0]))


def test_wgs84_nadir_from_iss_hits() -> None:
    """A nadir ECEF ray from LEO intersects the ellipsoid in front of the camera."""
    a = 6378137.0
    f = 0.0033528106647474805
    r0 = np.array([a + 400_000.0, 0.0, 0.0])
    d = np.array([-1.0, 0.0, 0.0])
    hit = wgs84_intersect(r0, d, a, f)
    assert hit is not None
    point, slant = hit
    assert slant > 1.0
    assert float(np.linalg.norm(point)) < a + 1.0
