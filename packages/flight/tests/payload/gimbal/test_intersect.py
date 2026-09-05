"""Tests for pinhole CoG Earth intersect."""

import math

from flight.libs.config import EphemerisConfig, SensorConfig
from flight.payload.gimbal.intersect import intersect_cog


def _iss_at_epoch() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Circular-orbit-like ECI state along +X with along-track +Y."""
    r = 6_378_137.0 + 400_000.0
    v = math.sqrt(3.986004418e14 / r)
    return (r, 0.0, 0.0), (0.0, v, 0.0)


def test_nadir_pixel_hits_earth() -> None:
    """A boresight pixel at nadir elevation intersects WGS-84."""
    eph = EphemerisConfig()
    sensor = SensorConfig()
    r_iss, v_iss = _iss_at_epoch()
    result = intersect_cog(
        p_cog_px=(sensor.width_px / 4.0, sensor.height_px / 4.0),
        theta_g_rad=0.0,
        r_iss_eci_m=r_iss,
        v_iss_eci_m_s=v_iss,
        utc_s=eph.epoch_utc_s,
        epoch_utc_s=eph.epoch_utc_s,
        omega_earth_rad_s=eph.omega_earth_rad_s,
        wgs84_a_m=eph.wgs84_a_m,
        wgs84_f=eph.wgs84_f,
        plane_width_px=sensor.width_px // 2,
        plane_height_px=sensor.height_px // 2,
        pixel_pitch_m=2.0 * sensor.pixel_um * 1.0e-6,
        focal_m=sensor.focal_length_mm * 1.0e-3,
        last_r_cog_ecef_m=None,
    )
    assert result.hit is True
    assert result.r_cog_ecef_m is not None
    assert result.slant_m > 1.0e5


def test_miss_keeps_last_cog() -> None:
    """A skyward look keeps the previous ECEF CoG and reports hit=False."""
    eph = EphemerisConfig()
    last = (1.0e6, 2.0e6, 3.0e6)
    r_iss, v_iss = _iss_at_epoch()
    result = intersect_cog(
        p_cog_px=(612.0, 512.0),
        theta_g_rad=math.radians(179.0),
        r_iss_eci_m=r_iss,
        v_iss_eci_m_s=v_iss,
        utc_s=eph.epoch_utc_s,
        epoch_utc_s=eph.epoch_utc_s,
        omega_earth_rad_s=eph.omega_earth_rad_s,
        wgs84_a_m=eph.wgs84_a_m,
        wgs84_f=eph.wgs84_f,
        plane_width_px=1224,
        plane_height_px=1024,
        pixel_pitch_m=6.9e-6,
        focal_m=0.150,
        last_r_cog_ecef_m=last,
    )
    assert result.hit is False
    assert result.r_cog_ecef_m == last
