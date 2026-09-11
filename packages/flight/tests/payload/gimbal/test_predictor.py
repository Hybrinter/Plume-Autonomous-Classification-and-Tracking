"""Tests for the co-rotating elevation and optical-azimuth predictor."""

import math

import numpy as np
from flight.hal.drivers_sim import SimIssEphemeris
from flight.libs.config import EphemerisConfig
from flight.libs.time import ManualClock
from flight.libs.types import Ok
from flight.payload.gimbal.geo import ecef_from_eci, eci_from_ecef, lvlh_axes
from flight.payload.gimbal.predictor import predict_los


def _nadir_cog(r_iss: tuple[float, float, float], wgs84_a_m: float) -> tuple[float, float, float]:
    """Scale ISS ECI position onto the equatorial radius (near-nadir ECEF at epoch)."""
    r_norm = math.hypot(r_iss[0], r_iss[1], r_iss[2])
    scale = wgs84_a_m / r_norm
    return (r_iss[0] * scale, r_iss[1] * scale, r_iss[2] * scale)


def _optical_az(
    utc_s: float,
    r_iss_eci_m: tuple[float, float, float],
    v_iss_eci_m_s: tuple[float, float, float],
    r_cog_ecef_m: tuple[float, float, float],
    omega_earth_rad_s: float,
    epoch_utc_s: float,
) -> float:
    """Optical azimuth atan2(ly, hypot(lx, lz)) of a frozen ECEF CoG."""
    r_iss = np.asarray(r_iss_eci_m, dtype=np.float64)
    v_iss = np.asarray(v_iss_eci_m_s, dtype=np.float64)
    r_cog_ecef = np.asarray(r_cog_ecef_m, dtype=np.float64)
    r_cog_eci = eci_from_ecef(r_cog_ecef, omega_earth_rad_s, utc_s, epoch_utc_s)
    look = r_cog_eci - r_iss
    x_hat, y_hat, z_hat = lvlh_axes(r_iss, v_iss)
    lx = float(look @ x_hat)
    ly = float(look @ y_hat)
    lz = float(look @ z_hat)
    return math.atan2(ly, math.hypot(lx, lz))


def test_frozen_ecef_matches_theta_finite_difference() -> None:
    """omega_el matches a central difference of theta_el at a frozen ECEF CoG."""
    eph = EphemerisConfig()
    clock = ManualClock(utc_s=eph.epoch_utc_s)
    sim = SimIssEphemeris(clock, eph)
    t0 = eph.epoch_utc_s
    state0 = sim.read_state(t0)
    assert isinstance(state0, Ok)
    r_iss = state0.value.r_m
    r_cog = _nadir_cog(r_iss, eph.wgs84_a_m)
    dt = 0.05
    theta0, omega, _omega_az = predict_los(
        t0, r_iss, state0.value.v_m_s, r_cog, eph.omega_earth_rad_s, eph.epoch_utc_s
    )
    plus = sim.read_state(t0 + dt)
    minus = sim.read_state(t0 - dt)
    assert isinstance(plus, Ok) and isinstance(minus, Ok)
    theta_p, _, _ = predict_los(
        t0 + dt,
        plus.value.r_m,
        plus.value.v_m_s,
        r_cog,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    theta_m, _, _ = predict_los(
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


def test_omega_az_matches_optical_azimuth_finite_difference() -> None:
    """omega_az matches a central difference of atan2(ly, hypot(lx, lz))."""
    eph = EphemerisConfig()
    sim = SimIssEphemeris(ManualClock(utc_s=eph.epoch_utc_s), eph)
    t0 = eph.epoch_utc_s
    state0 = sim.read_state(t0)
    assert isinstance(state0, Ok)
    r_iss = state0.value.r_m
    r_cog = _nadir_cog(r_iss, eph.wgs84_a_m)
    dt = 0.05
    _theta, _omega_el, omega_az = predict_los(
        t0, r_iss, state0.value.v_m_s, r_cog, eph.omega_earth_rad_s, eph.epoch_utc_s
    )
    plus = sim.read_state(t0 + dt)
    minus = sim.read_state(t0 - dt)
    assert isinstance(plus, Ok) and isinstance(minus, Ok)
    az_p = _optical_az(
        t0 + dt,
        plus.value.r_m,
        plus.value.v_m_s,
        r_cog,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    az_m = _optical_az(
        t0 - dt,
        minus.value.r_m,
        minus.value.v_m_s,
        r_cog,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    fd = (az_p - az_m) / (2.0 * dt)
    assert abs(omega_az - fd) / max(abs(fd), 1e-9) < 0.05


def test_omega_az_near_zero_when_earth_rotation_off() -> None:
    """Omega_E = 0 yields |omega_az| near 0 at a nadir CoG."""
    eph = EphemerisConfig()
    sim = SimIssEphemeris(ManualClock(utc_s=eph.epoch_utc_s), eph)
    t0 = eph.epoch_utc_s
    state0 = sim.read_state(t0)
    assert isinstance(state0, Ok)
    r_iss = state0.value.r_m
    r_cog = _nadir_cog(r_iss, eph.wgs84_a_m)
    _theta, _omega_el, omega_az = predict_los(
        t0, r_iss, state0.value.v_m_s, r_cog, 0.0, eph.epoch_utc_s
    )
    assert abs(omega_az) < 1e-9


def test_earth_rotation_changes_omega_el() -> None:
    """Omega_E on versus off changes omega_el at a nadir CoG."""
    eph = EphemerisConfig()
    sim = SimIssEphemeris(ManualClock(utc_s=eph.epoch_utc_s), eph)
    t0 = eph.epoch_utc_s
    state0 = sim.read_state(t0)
    assert isinstance(state0, Ok)
    r_iss = state0.value.r_m
    v_iss = state0.value.v_m_s
    r_cog = _nadir_cog(r_iss, eph.wgs84_a_m)
    _th_on, el_on, _az_on = predict_los(
        t0, r_iss, v_iss, r_cog, eph.omega_earth_rad_s, eph.epoch_utc_s
    )
    _th_off, el_off, _az_off = predict_los(t0, r_iss, v_iss, r_cog, 0.0, eph.epoch_utc_s)
    assert abs(el_on - el_off) > 1e-8


def test_omega_az_changes_with_earth_rotation_at_nadir() -> None:
    """Omega_E on versus off changes omega_az at a nadir CoG."""
    eph = EphemerisConfig()
    sim = SimIssEphemeris(ManualClock(utc_s=eph.epoch_utc_s), eph)
    t0 = eph.epoch_utc_s
    state0 = sim.read_state(t0)
    assert isinstance(state0, Ok)
    r_iss = state0.value.r_m
    v_iss = state0.value.v_m_s
    r_cog = _nadir_cog(r_iss, eph.wgs84_a_m)
    _th_on, _el_on, az_on = predict_los(
        t0, r_iss, v_iss, r_cog, eph.omega_earth_rad_s, eph.epoch_utc_s
    )
    _th_off, _el_off, az_off = predict_los(t0, r_iss, v_iss, r_cog, 0.0, eph.epoch_utc_s)
    assert abs(az_on - az_off) > 1e-8


def test_equator_omega_az_exceeds_high_latitude_due_east() -> None:
    """Equator-node ISS has larger |omega_az| than due-east motion at high latitude."""
    eph = EphemerisConfig()
    t0 = eph.epoch_utc_s
    r_eq = 6_778_137.0
    v_eq = math.sqrt(eph.mu_m3_s2 / r_eq)
    inc = math.radians(eph.inclination_deg)
    r_iss_eq = (r_eq, 0.0, 0.0)
    v_iss_eq = (0.0, v_eq * math.cos(inc), v_eq * math.sin(inc))
    r_cog_eq = (eph.wgs84_a_m, 0.0, 0.0)
    _th_eq, _el_eq, az_eq = predict_los(
        t0, r_iss_eq, v_iss_eq, r_cog_eq, eph.omega_earth_rad_s, eph.epoch_utc_s
    )

    r_hi = (0.0, 0.0, r_eq)
    v_iss_hi = (0.0, v_eq, 0.0)
    b_m = eph.wgs84_a_m * (1.0 - eph.wgs84_f)
    r_cog_hi = (0.0, 0.0, b_m)
    _th_hi, _el_hi, az_hi = predict_los(
        t0, r_hi, v_iss_hi, r_cog_hi, eph.omega_earth_rad_s, eph.epoch_utc_s
    )
    assert abs(az_eq) > abs(az_hi)
