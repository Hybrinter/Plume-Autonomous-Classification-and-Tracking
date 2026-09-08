"""Tests for mount / LVLH / WGS-84 geometry helpers."""

import math

import numpy as np
from flight.libs.config import EphemerisConfig
from flight.payload.gimbal.geo import (
    boresight_mount,
    cam_ray_to_mount,
    ecef_from_eci,
    lvlh_axes,
    mount_to_eci,
    pinhole_cam_ray,
    ry,
    wgs84_intersect,
)


def test_boresight_matches_ry_nadir() -> None:
    """Boresight (sin theta, 0, cos theta) is R_y(theta) applied to nadir +z."""
    theta = math.radians(20.0)
    b = boresight_mount(theta)
    mapped = ry(theta) @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(b, mapped)
    assert abs(b[1]) < 1e-15
    assert b[0] > 0.0


def test_lvlh_matches_analysis_triad() -> None:
    """+x along-track, +y starboard (-h), +z nadir, and x × y = z."""
    r = np.array([6_778_137.0, 0.0, 0.0])
    v = np.array([0.0, 7_600.0, 0.0])
    x_hat, y_hat, z_hat = lvlh_axes(r, v)
    h = np.cross(r, v)
    assert np.allclose(z_hat, -r / np.linalg.norm(r))
    assert np.allclose(y_hat, -h / np.linalg.norm(h))
    assert np.allclose(np.cross(x_hat, y_hat), z_hat, atol=1e-12)


def test_image_right_is_mount_starboard() -> None:
    """Camera +X (image right) maps to mount +y (starboard) at nadir."""
    ray = pinhole_cam_ray(662.0, 512.0, 1224, 1024, 6.9e-6, 0.150)
    assert ray[0] > 0.0
    mount = cam_ray_to_mount(ray, 0.0)
    assert mount[1] > 0.0


def test_image_right_cog_is_starboard_of_ground_track() -> None:
    """An image-right CoG ECEF point lies starboard of the nadir ground track."""
    eph = EphemerisConfig()
    r_eci = np.array([eph.wgs84_a_m + 400_000.0, 0.0, 0.0], dtype=np.float64)
    v_eci = np.array([0.0, math.sqrt(eph.mu_m3_s2 / float(np.linalg.norm(r_eci))), 0.0])
    d_cam = pinhole_cam_ray(712.0, 512.0, 1224, 1024, 6.9e-6, 0.150)
    d_mount = cam_ray_to_mount(d_cam, 0.0)
    d_eci = mount_to_eci(d_mount, r_eci, v_eci)
    utc = eph.epoch_utc_s
    r_ecef = ecef_from_eci(r_eci, eph.omega_earth_rad_s, utc, utc)
    d_ecef = ecef_from_eci(d_eci, eph.omega_earth_rad_s, utc, utc)
    nadir_ecef = ecef_from_eci(-r_eci / np.linalg.norm(r_eci), eph.omega_earth_rad_s, utc, utc)
    cog = wgs84_intersect(r_ecef, d_ecef, eph.wgs84_a_m, eph.wgs84_f)
    nadir = wgs84_intersect(r_ecef, nadir_ecef, eph.wgs84_a_m, eph.wgs84_f)
    assert cog is not None and nadir is not None
    offset = cog[0] - nadir[0]
    _x_hat, y_hat, _z_hat = lvlh_axes(r_eci, v_eci)
    y_ecef = ecef_from_eci(y_hat, eph.omega_earth_rad_s, utc, utc)
    assert float(offset @ y_ecef) > 0.0


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
