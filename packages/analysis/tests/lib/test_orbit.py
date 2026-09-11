"""Facade checks: analysis orbit km wrappers match sim SI models."""

from __future__ import annotations

import math

import numpy as np
from analysis.lib.constants import MU_KM3_S2, WGS84_A_KM, WGS84_B_KM
from analysis.lib.orbit import (
    IssTle,
    argument_of_latitude,
    build_orbit,
    iss_eci,
    wgs84_geocentric_radius_km,
)
from sim.environment.models.earth import geocentric_radius_m
from sim.environment.models.orbit import CircularKepler


def test_geocentric_radius_matches_sim() -> None:
    """km radius is sim geocentric_radius_m / 1000."""
    lat = math.radians(51.6312)
    a_m = WGS84_A_KM * 1000.0
    f = 1.0 - WGS84_B_KM / WGS84_A_KM
    assert wgs84_geocentric_radius_km(lat) == geocentric_radius_m(a_m, f, lat) / 1000.0


def test_iss_eci_matches_circular_kepler() -> None:
    """iss_eci is CircularKepler in kilometres with argument-of-latitude phase."""
    tle = IssTle(
        epoch="2026-09-01",
        inclination_deg=51.6312,
        eccentricity=0.0005055,
        mean_motion_rev_per_day=15.48958602,
    )
    orbit = build_orbit(tle, use_perigee=False)
    u0 = argument_of_latitude(51.6312, orbit.inclination_rad)
    t_s = 120.0
    pos_km, vel_km_s = iss_eci(t_s, orbit, u0)
    kepler = CircularKepler(
        inclination_deg=orbit.inclination_deg,
        mean_motion_rev_per_day=orbit.n_rad_s * 86400.0 / (2.0 * math.pi),
        mu_m3_s2=MU_KM3_S2 * 1.0e9,
        epoch_utc_s=0.0,
    )
    state = kepler.state_eci(t_s + u0 / orbit.n_rad_s)
    np.testing.assert_allclose(pos_km, np.array(state.r_m) / 1000.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(vel_km_s, np.array(state.v_m_s) / 1000.0, rtol=0.0, atol=1e-12)
