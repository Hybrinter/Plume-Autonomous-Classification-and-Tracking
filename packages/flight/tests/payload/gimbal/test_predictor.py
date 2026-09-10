"""Tests for the co-rotating elevation predictor."""

import math

import numpy as np
from flight.hal.drivers_sim import SimIssEphemeris
from flight.libs.config import EphemerisConfig
from flight.libs.time import ManualClock
from flight.libs.types import Ok
from flight.payload.gimbal.geo import ecef_from_eci
from flight.payload.gimbal.predictor import predict_los


def test_frozen_ecef_matches_theta_finite_difference() -> None:
    """omega_t_nom matches a central difference of theta_los at a frozen ECEF CoG."""
    eph = EphemerisConfig()
    clock = ManualClock(utc_s=eph.epoch_utc_s)
    sim = SimIssEphemeris(clock, eph)
    t0 = eph.epoch_utc_s
    state0 = sim.read_state(t0)
    assert isinstance(state0, Ok)
    r_iss = state0.value.r_m
    # Nadir ECEF at epoch (frames aligned).
    r_norm = math.hypot(r_iss[0], r_iss[1], r_iss[2])
    scale = eph.wgs84_a_m / r_norm
    r_cog = (r_iss[0] * scale, r_iss[1] * scale, r_iss[2] * scale)
    dt = 0.05
    theta0, omega = predict_los(
        t0, r_iss, state0.value.v_m_s, r_cog, eph.omega_earth_rad_s, eph.epoch_utc_s
    )
    plus = sim.read_state(t0 + dt)
    minus = sim.read_state(t0 - dt)
    assert isinstance(plus, Ok) and isinstance(minus, Ok)
    theta_p, _ = predict_los(
        t0 + dt,
        plus.value.r_m,
        plus.value.v_m_s,
        r_cog,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    theta_m, _ = predict_los(
        t0 - dt,
        minus.value.r_m,
        minus.value.v_m_s,
        r_cog,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    fd = (theta_p - theta_m) / (2.0 * dt)
    assert abs(omega - fd) / max(abs(fd), 1e-9) < 0.05
    assert abs(theta0) < math.radians(2.0)


def test_earth_rotation_rotates_ecef_into_eci() -> None:
    """ecef_from_eci at a later UTC is not the identity once Earth has rotated."""
    eph = EphemerisConfig()
    vec = (1.0, 0.0, 0.0)
    later = eph.epoch_utc_s + 3600.0
    rotated = ecef_from_eci(
        np.asarray(vec),
        eph.omega_earth_rad_s,
        later,
        eph.epoch_utc_s,
    )
    assert abs(float(rotated[0]) - 1.0) > 1e-4
