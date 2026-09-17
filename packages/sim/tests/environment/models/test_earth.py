"""Tests for sim.environment.models.earth."""

import math

from flight.libs.config import EphemerisConfig
from sim.environment.models.earth import geocentric_radius_m


def test_geocentric_radius_m_equator_is_semi_major() -> None:
    """Equatorial geocentric radius equals WGS-84 semi-major axis."""
    cfg = EphemerisConfig()
    assert geocentric_radius_m(cfg.wgs84_a_m, cfg.wgs84_f, 0.0) == cfg.wgs84_a_m


def test_geocentric_radius_m_pole_is_semi_minor() -> None:
    """Polar geocentric radius equals WGS-84 semi-minor axis."""
    cfg = EphemerisConfig()
    b_m = cfg.wgs84_a_m * (1.0 - cfg.wgs84_f)
    got = geocentric_radius_m(cfg.wgs84_a_m, cfg.wgs84_f, math.pi / 2.0)
    assert abs(got - b_m) < 1e-6


def test_geocentric_radius_m_45_deg() -> None:
    """45 deg geocentric latitude matches the radial-intersection value."""
    cfg = EphemerisConfig()
    lat_rad = math.radians(45.0)
    expected = 6367417.7249666825
    got = geocentric_radius_m(cfg.wgs84_a_m, cfg.wgs84_f, lat_rad)
    assert abs(got - expected) < 1e-6


def test_geocentric_radius_m_satisfies_ellipsoid_equation() -> None:
    """Intermediate latitude lies on the WGS-84 ellipsoid."""
    cfg = EphemerisConfig()
    a_m = cfg.wgs84_a_m
    b_m = a_m * (1.0 - cfg.wgs84_f)
    lat_rad = math.radians(30.0)
    r_m = geocentric_radius_m(a_m, cfg.wgs84_f, lat_rad)
    c_lat, s_lat = math.cos(lat_rad), math.sin(lat_rad)
    ellipsoid = (r_m * c_lat / a_m) ** 2 + (r_m * s_lat / b_m) ** 2
    assert abs(ellipsoid - 1.0) < 1e-12
