"""Tests for scene-point selection and residual-reference identity."""

import math

from flight.libs.config import EphemerisConfig, PredictorConfig
from flight.libs.types import GimbalState
from flight.payload.gimbal.intersect import intersect_boresight
from flight.payload.gimbal.predictor import predict_los
from flight.payload.gimbal.scene import (
    SceneEstimate,
    SceneSource,
    acquire_resets_residual,
    select_scene,
)


def _iss() -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Circular-LEO ECI state at the ephemeris epoch."""
    eph = EphemerisConfig()
    radius = 6_378_137.0 + 400_000.0
    speed = math.sqrt(eph.mu_m3_s2 / radius)
    return (radius, 0.0, 0.0), (0.0, speed, 0.0), eph.epoch_utc_s


def _scene(
    mode: GimbalState,
    r_cog_ecef_m: tuple[float, float, float] | None,
    *,
    with_iss: bool = True,
    theta_g_rad: float = 0.0,
) -> SceneEstimate:
    """select_scene with default ephemeris and height-proxy constants."""
    eph = EphemerisConfig()
    r_iss, v_iss, utc = _iss()
    return select_scene(
        mode,
        r_cog_ecef_m,
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


def test_tracking_cog_with_iss_is_cog_source() -> None:
    """TRACKING with a stored CoG and ISS predicts from that CoG."""
    eph = EphemerisConfig()
    cog = (eph.wgs84_a_m, 0.0, 0.0)
    r_iss, v_iss, utc = _iss()
    scene = _scene(GimbalState.TRACKING, cog, theta_g_rad=math.radians(10.0))
    assert scene.source is SceneSource.COG
    assert scene.point_ecef_m == cog
    assert scene.nav_valid is True
    assert scene.t_utc_s == utc
    assert scene.los is not None
    expected = predict_los(utc, r_iss, v_iss, cog, eph.omega_earth_rad_s, eph.epoch_utc_s)
    assert abs(scene.los.elevation_rate_rad_s - expected.elevation_rate_rad_s) < 1e-12


def test_tracking_without_iss_is_unknown_nav_not_zero_rate() -> None:
    """Missing ISS is unknown navigation. It is not a LosPrediction of 0.0."""
    eph = EphemerisConfig()
    cog = (eph.wgs84_a_m, 0.0, 0.0)
    scene = _scene(GimbalState.TRACKING, cog, with_iss=False)
    assert scene.source is SceneSource.COG
    assert scene.point_ecef_m == cog
    assert scene.nav_valid is False
    assert scene.t_utc_s is None
    assert scene.los is None


def test_rewind_uses_boresight_not_stored_cog() -> None:
    """REWIND selects the boresight hit. It does not keep a lost-plume CoG."""
    eph = EphemerisConfig()
    cog = (eph.wgs84_a_m, 0.0, 0.0)
    theta_g = math.radians(20.0)
    r_iss, v_iss, utc = _iss()
    scene = _scene(GimbalState.REWIND, cog, theta_g_rad=theta_g)
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
    assert scene.point_ecef_m != cog
    assert scene.nav_valid is True
    assert scene.los is not None
    omega_bore = predict_los(
        utc, r_iss, v_iss, bore.point_ecef_m, eph.omega_earth_rad_s, eph.epoch_utc_s
    ).elevation_rate_rad_s
    omega_cog = predict_los(
        utc, r_iss, v_iss, cog, eph.omega_earth_rad_s, eph.epoch_utc_s
    ).elevation_rate_rad_s
    assert abs(scene.los.elevation_rate_rad_s - omega_bore) < 1e-12
    assert abs(omega_bore - omega_cog) > 1e-8


def test_tracking_without_cog_is_none_source() -> None:
    """TRACKING with no CoG has no Earth point even when ISS is present."""
    scene = _scene(GimbalState.TRACKING, None, with_iss=True)
    assert scene.source is SceneSource.NONE
    assert scene.point_ecef_m is None
    assert scene.los is None
    assert scene.nav_valid is True


def test_safe_has_no_scene_point() -> None:
    """SAFE selects no Earth point."""
    eph = EphemerisConfig()
    scene = _scene(GimbalState.SAFE, (eph.wgs84_a_m, 0.0, 0.0))
    assert scene.source is SceneSource.NONE
    assert scene.point_ecef_m is None
    assert scene.los is None


def test_acquire_resets_on_cold_first_blob() -> None:
    """The first blob from a cold aggregate resets the residual."""
    assert acquire_resets_residual(
        previous_mode=GimbalState.TRACKING,
        new_mode=GimbalState.TRACKING,
        previous_aggregate_live=False,
        previous_blob_ids=frozenset(),
        new_blob_ids=frozenset({2}),
    )


def test_acquire_resets_from_rewind() -> None:
    """A blob that enters TRACKING from REWIND always resets the residual."""
    assert acquire_resets_residual(
        previous_mode=GimbalState.REWIND,
        new_mode=GimbalState.TRACKING,
        previous_aggregate_live=False,
        previous_blob_ids=frozenset({2}),
        new_blob_ids=frozenset({2}),
    )


def test_acquire_resets_on_unmatched_blob_ids() -> None:
    """A TRACKING blob set with no overlapping blob_id is a new object."""
    assert acquire_resets_residual(
        previous_mode=GimbalState.TRACKING,
        new_mode=GimbalState.TRACKING,
        previous_aggregate_live=True,
        previous_blob_ids=frozenset({2}),
        new_blob_ids=frozenset({3}),
    )


def test_acquire_keeps_residual_on_iou_match() -> None:
    """Overlapping blob IDs while already TRACKING keep the residual."""
    assert not acquire_resets_residual(
        previous_mode=GimbalState.TRACKING,
        new_mode=GimbalState.TRACKING,
        previous_aggregate_live=True,
        previous_blob_ids=frozenset({2}),
        new_blob_ids=frozenset({2}),
    )


def test_acquire_keeps_residual_on_single_miss() -> None:
    """An empty frame while still TRACKING does not reset the residual."""
    assert not acquire_resets_residual(
        previous_mode=GimbalState.TRACKING,
        new_mode=GimbalState.TRACKING,
        previous_aggregate_live=True,
        previous_blob_ids=frozenset({2}),
        new_blob_ids=frozenset(),
    )


def test_acquire_keeps_residual_after_miss_cleared_ids() -> None:
    """A reappearing blob after an empty previous set is still a coast, not a new object."""
    assert not acquire_resets_residual(
        previous_mode=GimbalState.TRACKING,
        new_mode=GimbalState.TRACKING,
        previous_aggregate_live=True,
        previous_blob_ids=frozenset(),
        new_blob_ids=frozenset({2}),
    )
