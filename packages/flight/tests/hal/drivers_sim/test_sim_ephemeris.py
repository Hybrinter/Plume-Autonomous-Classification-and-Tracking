"""Tests for the circular Keplerian sim ISS ephemeris."""

import math

from flight.hal.drivers_sim import SimIssEphemeris
from flight.libs.config import EphemerisConfig
from flight.libs.time import ManualClock
from flight.libs.types import Ok


def test_read_state_is_eci_meters() -> None:
    """read_state returns ECI position and velocity in meters."""
    cfg = EphemerisConfig()
    eph = SimIssEphemeris(ManualClock(utc_s=cfg.epoch_utc_s), cfg)
    result = eph.read_state(cfg.epoch_utc_s)
    assert isinstance(result, Ok)
    assert result.value.frame == "ECI"
    r = result.value.r_m
    v = result.value.v_m_s
    r_norm = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
    n = cfg.mean_motion_rev_per_day * 2.0 * math.pi / 86400.0
    a = (cfg.mu_m3_s2 / (n * n)) ** (1.0 / 3.0)
    assert abs(r_norm - a) / a < 1e-9
    v_norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    assert abs(v_norm - n * a) / (n * a) < 1e-9
