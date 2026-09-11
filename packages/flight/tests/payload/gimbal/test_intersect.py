"""Tests for pinhole CoG and boresight Earth intersect."""

import math

from flight.libs.config import EphemerisConfig, SensorConfig
from flight.payload.gimbal.intersect import IntersectResult, intersect_boresight, intersect_cog
from flight.payload.gimbal.predictor import predict_los


def _iss_at_epoch() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Circular-orbit-like ECI state along +X with along-track +Y."""
    r = 6_378_137.0 + 400_000.0
    v = math.sqrt(3.986004418e14 / r)
    return (r, 0.0, 0.0), (0.0, v, 0.0)


def _cog(
    p_cog_px: tuple[float, float],
    theta_g_rad: float,
    last_r_cog_ecef_m: tuple[float, float, float] | None,
    height_m: float,
) -> IntersectResult:
    """intersect_cog at epoch with default sensor optics."""
    eph = EphemerisConfig()
    sensor = SensorConfig()
    r_iss, v_iss = _iss_at_epoch()
    return intersect_cog(
        p_cog_px=p_cog_px,
        theta_g_rad=theta_g_rad,
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
        last_r_cog_ecef_m=last_r_cog_ecef_m,
        height_m=height_m,
    )


def _boresight(
    theta_g_rad: float,
    last_r_cog_ecef_m: tuple[float, float, float] | None,
    height_m: float,
) -> IntersectResult:
    """intersect_boresight at epoch."""
    eph = EphemerisConfig()
    r_iss, v_iss = _iss_at_epoch()
    return intersect_boresight(
        theta_g_rad=theta_g_rad,
        r_iss_eci_m=r_iss,
        v_iss_eci_m_s=v_iss,
        utc_s=eph.epoch_utc_s,
        epoch_utc_s=eph.epoch_utc_s,
        omega_earth_rad_s=eph.omega_earth_rad_s,
        wgs84_a_m=eph.wgs84_a_m,
        wgs84_f=eph.wgs84_f,
        last_r_cog_ecef_m=last_r_cog_ecef_m,
        height_m=height_m,
    )


def test_nadir_pixel_hits_earth() -> None:
    """A boresight pixel at nadir elevation intersects the height ellipsoid."""
    sensor = SensorConfig()
    result = _cog(
        p_cog_px=(sensor.width_px / 4.0, sensor.height_px / 4.0),
        theta_g_rad=0.0,
        last_r_cog_ecef_m=None,
        height_m=2000.0,
    )
    assert result.hit is True
    assert result.r_cog_ecef_m is not None
    assert result.slant_m > 1.0e5


def test_miss_keeps_last_cog() -> None:
    """A skyward look keeps the previous ECEF CoG and reports hit=False."""
    last = (1.0e6, 2.0e6, 3.0e6)
    result = _cog(
        p_cog_px=(612.0, 512.0),
        theta_g_rad=math.radians(179.0),
        last_r_cog_ecef_m=last,
        height_m=0.0,
    )
    assert result.hit is False
    assert result.r_cog_ecef_m == last


def test_height_proxy_changes_ecef_hit() -> None:
    """The same nadir ray hits a larger radius on the 2 km ellipsoid than on the surface."""
    sensor = SensorConfig()
    p_cog = (sensor.width_px / 4.0, sensor.height_px / 4.0)
    surface = _cog(p_cog, 0.0, None, 0.0)
    raised = _cog(p_cog, 0.0, None, 2000.0)
    assert surface.hit is True and raised.hit is True
    assert surface.r_cog_ecef_m is not None and raised.r_cog_ecef_m is not None
    r_surf = math.hypot(*surface.r_cog_ecef_m)
    r_hi = math.hypot(*raised.r_cog_ecef_m)
    assert r_hi > r_surf
    assert abs((r_hi - r_surf) - 2000.0) < 50.0


def test_intersect_boresight_nadir_hits() -> None:
    """A principal-point ray at nadir elevation intersects the height ellipsoid."""
    result = _boresight(theta_g_rad=0.0, last_r_cog_ecef_m=None, height_m=2000.0)
    assert result.hit is True
    assert result.r_cog_ecef_m is not None
    assert result.slant_m > 1.0e5


def test_height_proxy_changes_omega_el_at_large_look() -> None:
    """Surface vs 2 km lock changes omega_el at a large elevation."""
    eph = EphemerisConfig()
    sensor = SensorConfig()
    r_iss, v_iss = _iss_at_epoch()
    p_cog = (sensor.width_px / 4.0, sensor.height_px / 4.0)
    theta = math.radians(35.0)
    surface = _cog(p_cog, theta, None, 0.0)
    raised = _cog(p_cog, theta, None, 2000.0)
    assert surface.hit is True and raised.hit is True
    assert surface.r_cog_ecef_m is not None and raised.r_cog_ecef_m is not None
    _th0, w0, _az0 = predict_los(
        eph.epoch_utc_s,
        r_iss,
        v_iss,
        surface.r_cog_ecef_m,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    _th2, w2, _az2 = predict_los(
        eph.epoch_utc_s,
        r_iss,
        v_iss,
        raised.r_cog_ecef_m,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    assert abs(w0 - w2) > 1e-8


def test_intersect_boresight_miss_keeps_last() -> None:
    """A skyward boresight keeps the previous ECEF point and reports hit=False."""
    last = (1.0e6, 2.0e6, 3.0e6)
    result = _boresight(theta_g_rad=math.radians(179.0), last_r_cog_ecef_m=last, height_m=2000.0)
    assert result.hit is False
    assert result.r_cog_ecef_m == last
