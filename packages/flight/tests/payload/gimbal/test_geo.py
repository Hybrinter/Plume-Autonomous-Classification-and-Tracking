"""Tests for mount / LVLH / WGS-84 geometry helpers."""

import math

import numpy as np
from flight.libs.config import EphemerisConfig
from flight.payload.gimbal.geo import (
    boresight_mount,
    cam_ray_to_mount,
    ecef_from_eci,
    height_proxy_semiaxes,
    lvlh_axes,
    mount_to_eci,
    pinhole_cam_ray,
    ry,
    wgs84_intersect,
    wgs84_intersect_at_height,
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


def test_height_intersect_is_farther_than_surface_at_nadir() -> None:
    """A nadir height-ellipsoid hit lies about height_m farther out than the surface."""
    a = 6378137.0
    f = 0.0033528106647474805
    height_m = 2000.0
    r0 = np.array([a + 400_000.0, 0.0, 0.0])
    d = np.array([-1.0, 0.0, 0.0])
    surface = wgs84_intersect(r0, d, a, f)
    raised = wgs84_intersect_at_height(r0, d, a, f, height_m)
    assert surface is not None and raised is not None
    r_surf = float(np.linalg.norm(surface[0]))
    r_hi = float(np.linalg.norm(raised[0]))
    assert abs((r_hi - r_surf) - height_m) < 1.0


def test_height_proxy_semiaxes_inflate_a_and_b() -> None:
    """The 2 km proxy ellipsoid derives a' = a + h and b' = b + h."""
    a = 6378137.0
    f = 0.0033528106647474805
    height_m = 2000.0
    b = a * (1.0 - f)
    a_h, b_h = height_proxy_semiaxes(a, f, height_m)
    assert a_h == a + height_m
    assert b_h == b + height_m
    a0, b0 = height_proxy_semiaxes(a, f, 0.0)
    assert a0 == a and b0 == b
    f_h = 1.0 - b_h / a_h
    r0 = np.array([a + 400_000.0, 0.0, 0.0])
    d = np.array([-1.0, 0.0, 0.0])
    direct = wgs84_intersect(r0, d, a_h, f_h)
    proxy = wgs84_intersect_at_height(r0, d, a, f, height_m)
    assert direct is not None and proxy is not None
    assert np.allclose(direct[0], proxy[0])
    assert abs(direct[1] - proxy[1]) < 1e-9
