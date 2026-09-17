"""Orbit wrappers over sim CircularKepler and geocentric_radius_m."""

import math

import numpy as np
from analysis.studies.single_axis_vs_dual_axis_gimbal.assumptions import TLE
from analysis.studies.single_axis_vs_dual_axis_gimbal.constants import MU_KM3_S2
from analysis.studies.single_axis_vs_dual_axis_gimbal.orbit import (
    build_orbit,
    ephemeris_from_orbit,
    ephemeris_from_tle,
    iss_eci,
    wgs84_geocentric_radius_km,
)
from flight.libs.config import EphemerisConfig
from sim.environment.models.orbit import CircularKepler


def test_geocentric_radius_km_equator_is_wgs84_a() -> None:
    """Equatorial geocentric radius equals WGS-84 semi-major axis in km."""
    cfg = EphemerisConfig()
    expected_km = cfg.wgs84_a_m / 1000.0
    assert abs(wgs84_geocentric_radius_km(0.0) - expected_km) < 1e-12


def test_geocentric_radius_km_45_deg() -> None:
    """45 deg geocentric latitude matches the radial-intersection value in km."""
    lat_rad = math.radians(45.0)
    expected_km = 6367417.7249666825 / 1000.0
    assert abs(wgs84_geocentric_radius_km(lat_rad) - expected_km) < 1e-9


def test_iss_eci_wraps_circular_kepler() -> None:
    """t = 0, u0 = 0 matches CircularKepler from TLE ephemeris, in kilometres."""
    sma = build_orbit(TLE, use_perigee=False)
    pos_km, vel_km_s = iss_eci(0.0, sma, 0.0)
    eph = ephemeris_from_tle(TLE)
    kepler0 = CircularKepler(
        inclination_deg=eph.inclination_deg,
        mean_motion_rev_per_day=eph.mean_motion_rev_per_day,
        mu_m3_s2=MU_KM3_S2 * 1.0e9,
        epoch_utc_s=0.0,
    )
    node = kepler0.state_eci(0.0)
    assert np.allclose(pos_km, np.asarray(node.r_m) / 1000.0, atol=1e-9)
    assert np.allclose(vel_km_s, np.asarray(node.v_m_s) / 1000.0, atol=1e-9)


def test_ephemeris_from_orbit_uses_orbit_mean_motion() -> None:
    """A perigee Orbit does not reuse TLE mean motion."""
    sma = build_orbit(TLE, use_perigee=False)
    peri = build_orbit(TLE, use_perigee=True)
    eph_sma = ephemeris_from_orbit(sma)
    eph_peri = ephemeris_from_orbit(peri)
    eph_tle = ephemeris_from_tle(TLE)
    assert eph_peri.mean_motion_rev_per_day > eph_sma.mean_motion_rev_per_day
    assert eph_peri.mean_motion_rev_per_day != eph_tle.mean_motion_rev_per_day
    assert eph_peri.inclination_deg == peri.inclination_deg
