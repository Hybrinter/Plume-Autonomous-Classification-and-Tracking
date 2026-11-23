"""Tests for the cog_scene and boresight_scene prediction entry points."""

import math

from flight.libs.config import EphemerisConfig, PredictorConfig
from flight.payload.gimbal.intersect import intersect_boresight
from flight.payload.gimbal.predictor import predict_los
from flight.payload.gimbal.scene import (
    SceneEstimate,
    SceneSource,
    boresight_scene,
    cog_scene,
)


def _iss() -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Circular-LEO ECI state at the ephemeris epoch."""
    eph = EphemerisConfig()
    radius = 6_378_137.0 + 400_000.0
    speed = math.sqrt(eph.mu_m3_s2 / radius)
    return (radius, 0.0, 0.0), (0.0, speed, 0.0), eph.epoch_utc_s


def _cog_scene(
    r_cog_ecef_m: tuple[float, float, float] | None,
    *,
    with_iss: bool = True,
) -> SceneEstimate:
    """cog_scene with default ephemeris constants."""
    eph = EphemerisConfig()
    r_iss, v_iss, utc = _iss()
    return cog_scene(
        r_cog_ecef_m,
        r_iss if with_iss else None,
        v_iss if with_iss else None,
        utc if with_iss else None,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )


def _boresight_scene(
    *,
    with_iss: bool = True,
    theta_g_rad: float = 0.0,
) -> SceneEstimate:
    """boresight_scene with default ephemeris and height-proxy constants."""
    eph = EphemerisConfig()
    r_iss, v_iss, utc = _iss()
    return boresight_scene(
        r_iss if with_iss else None,
        v_iss if with_iss else None,
        utc if with_iss else None,
        theta_g_rad,
        PredictorConfig().cog_height_m,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
        eph.wgs84_a_m,
        eph.wgs84_f,
    )


def test_cog_scene_with_iss_is_cog_source() -> None:
    """A stored CoG with ISS predicts from that CoG."""
    eph = EphemerisConfig()
    cog = (eph.wgs84_a_m, 0.0, 0.0)
    r_iss, v_iss, utc = _iss()
    scene = _cog_scene(cog)
    assert scene.source is SceneSource.COG
    assert scene.point_ecef_m == cog
    assert scene.nav_valid is True
    assert scene.t_utc_s == utc
    assert scene.los is not None
    expected = predict_los(utc, r_iss, v_iss, cog, eph.omega_earth_rad_s, eph.epoch_utc_s)
    assert abs(scene.los.elevation_rate_rad_s - expected.elevation_rate_rad_s) < 1e-12


def test_cog_scene_without_iss_is_unknown_nav_not_zero_rate() -> None:
    """Missing ISS is unknown navigation. It is not a LosPrediction of 0.0."""
    eph = EphemerisConfig()
    cog = (eph.wgs84_a_m, 0.0, 0.0)
    scene = _cog_scene(cog, with_iss=False)
    assert scene.source is SceneSource.COG
    assert scene.point_ecef_m == cog
    assert scene.nav_valid is False
    assert scene.t_utc_s is None
    assert scene.los is None


def test_cog_scene_without_cog_is_none_source() -> None:
    """No stored CoG has no Earth point even when ISS is present."""
    scene = _cog_scene(None, with_iss=True)
    assert scene.source is SceneSource.NONE
    assert scene.point_ecef_m is None
    assert scene.los is None
    assert scene.nav_valid is True


def test_cog_scene_partial_nav_is_unknown() -> None:
    """A partial ISS state is unknown navigation, not a zero-rate scene."""
    eph = EphemerisConfig()
    cog = (eph.wgs84_a_m, 0.0, 0.0)
    r_iss, _v_iss, utc = _iss()
    scene = cog_scene(
        cog,
        r_iss,
        None,
        utc,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    assert scene.source is SceneSource.COG
    assert scene.point_ecef_m == cog
    assert scene.nav_valid is False
    assert scene.t_utc_s is None
    assert scene.los is None


def test_boresight_scene_uses_boresight_hit_not_a_stored_cog() -> None:
    """boresight_scene predicts the boresight intersect exactly."""
    eph = EphemerisConfig()
    theta_g = math.radians(20.0)
    r_iss, v_iss, utc = _iss()
    scene = _boresight_scene(theta_g_rad=theta_g)
    bore = intersect_boresight(
        theta_g,
        r_iss,
        v_iss,
        utc,
        eph.epoch_utc_s,
        eph.omega_earth_rad_s,
        eph.wgs84_a_m,
        eph.wgs84_f,
        PredictorConfig().cog_height_m,
    )
    assert bore is not None
    assert scene.source is SceneSource.BORESIGHT
    assert scene.point_ecef_m == bore.point_ecef_m
    assert scene.nav_valid is True
    assert scene.t_utc_s == utc
    assert scene.los is not None
    omega_bore = predict_los(
        utc, r_iss, v_iss, bore.point_ecef_m, eph.omega_earth_rad_s, eph.epoch_utc_s
    ).elevation_rate_rad_s
    assert abs(scene.los.elevation_rate_rad_s - omega_bore) < 1e-12


def test_boresight_scene_hit_differs_from_a_fixed_cog() -> None:
    """The boresight hit is not the stored target point."""
    eph = EphemerisConfig()
    cog = (eph.wgs84_a_m, 0.0, 0.0)
    theta_g = math.radians(20.0)
    r_iss, v_iss, utc = _iss()
    scene = _boresight_scene(theta_g_rad=theta_g)
    assert scene.point_ecef_m is not None
    assert scene.point_ecef_m != cog
    assert scene.los is not None
    omega_bore = scene.los.elevation_rate_rad_s
    omega_cog = predict_los(
        utc, r_iss, v_iss, cog, eph.omega_earth_rad_s, eph.epoch_utc_s
    ).elevation_rate_rad_s
    assert abs(omega_bore - omega_cog) > 1e-8


def test_boresight_scene_missed_ray_keeps_boresight_source() -> None:
    """A boresight ray that misses the ellipsoid has no point but BORESIGHT source."""
    eph = EphemerisConfig()
    theta_g = math.radians(-80.0)
    r_iss, v_iss, utc = _iss()
    bore = intersect_boresight(
        theta_g,
        r_iss,
        v_iss,
        utc,
        eph.epoch_utc_s,
        eph.omega_earth_rad_s,
        eph.wgs84_a_m,
        eph.wgs84_f,
        PredictorConfig().cog_height_m,
    )
    assert bore is None
    scene = _boresight_scene(theta_g_rad=theta_g)
    assert scene.source is SceneSource.BORESIGHT
    assert scene.point_ecef_m is None
    assert scene.nav_valid is True
    assert scene.los is None


def test_boresight_scene_without_iss_is_boresight_source_no_point() -> None:
    """Missing ISS still reports BORESIGHT source with unknown navigation."""
    scene = _boresight_scene(with_iss=False, theta_g_rad=math.radians(20.0))
    assert scene.source is SceneSource.BORESIGHT
    assert scene.point_ecef_m is None
    assert scene.nav_valid is False
    assert scene.t_utc_s is None
    assert scene.los is None
