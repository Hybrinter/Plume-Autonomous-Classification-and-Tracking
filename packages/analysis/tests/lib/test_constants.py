"""Facade checks: analysis constants match flight EphemerisConfig SI units."""

from analysis.lib.constants import MU_KM3_S2, OMEGA_EARTH_RAD_S, WGS84_A_KM, WGS84_B_KM
from flight.libs.config import EphemerisConfig


def test_constants_match_ephemeris_config() -> None:
    """km facades are the SI EphemerisConfig fields scaled by 1000."""
    eph = EphemerisConfig()
    assert MU_KM3_S2 == eph.mu_m3_s2 / 1.0e9
    assert OMEGA_EARTH_RAD_S == eph.omega_earth_rad_s
    assert WGS84_A_KM == eph.wgs84_a_m / 1000.0
    assert WGS84_B_KM == eph.wgs84_a_m * (1.0 - eph.wgs84_f) / 1000.0
