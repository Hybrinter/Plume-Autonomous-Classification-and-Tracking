"""Tests for named plume models, including PoissonLatitude draws."""

from __future__ import annotations

import math

import numpy as np
from flight.libs.config import EphemerisConfig, SensorConfig
from flight.libs.types import Err, Ok
from flight.payload.gimbal.geo import ecef_from_eci, height_proxy_semiaxes
from sim.environment import EnvironmentConfig, build_environment, camera_from_sensor
from sim.environment.config import PoissonLatitudeParams
from sim.environment.models.earth import Wgs84Ellipsoid
from sim.environment.models.orbit import CircularKepler
from sim.environment.models.plume import (
    _advance_iss_along_ground_track,
    _along_track_ahead_m,
    _nadir_hit_ecef,
    _place_along_track,
    _ssp_ground_separation_m,
    along_track_intensity_per_km,
)
from sim.environment.records import EnvTime, ShutterPose


def _shutter() -> ShutterPose:
    """Return a nadir shutter."""
    return ShutterPose(0.0, 0.0, 13.0, 0.0)


def test_zero_density_poisson_never_present() -> None:
    """A zero density table yields present=False and no CoG."""
    camera = camera_from_sensor(SensorConfig())
    cfg = EnvironmentConfig(
        plume="poisson_latitude",
        poisson_latitude=PoissonLatitudeParams(
            signed_lat_deg=(-90.0, 90.0),
            dens_per_km2=(0.0, 0.0),
        ),
    )
    built = build_environment(cfg, camera)
    assert isinstance(built, Ok)
    sample = built.value.evaluate(
        EnvTime(0.0, EphemerisConfig().epoch_utc_s),
        _shutter(),
        np.random.default_rng(0),
    )
    assert sample.truth.plume.present is False
    assert sample.truth.plume.cog_ecef_m is None
    assert sample.truth.look.visible is False


def test_poisson_draw_is_reproducible() -> None:
    """The same seed yields the same first CoG."""
    camera = camera_from_sensor(SensorConfig())
    cfg = EnvironmentConfig(
        plume="poisson_latitude",
        poisson_latitude=PoissonLatitudeParams(
            signed_lat_deg=(-90.0, 90.0),
            dens_per_km2=(1.0e-3, 1.0e-3),
            cross_track_half_km=5.0,
        ),
    )
    built = build_environment(cfg, camera)
    assert isinstance(built, Ok)
    env = built.value
    eph = EphemerisConfig()
    time = EnvTime(0.0, eph.epoch_utc_s)
    a = env.evaluate(time, _shutter(), np.random.default_rng(7))
    b = env.evaluate(time, _shutter(), np.random.default_rng(7))
    assert a.truth.plume.present is True
    assert a.truth.plume.cog_ecef_m == b.truth.plume.cog_ecef_m


def test_poisson_keeps_prior_while_ahead() -> None:
    """A second evaluate at the same time reuses the prior CoG without a new draw."""
    camera = camera_from_sensor(SensorConfig())
    cfg = EnvironmentConfig(
        plume="poisson_latitude",
        poisson_latitude=PoissonLatitudeParams(
            signed_lat_deg=(-90.0, 90.0),
            dens_per_km2=(1.0e-3, 1.0e-3),
        ),
    )
    built = build_environment(cfg, camera)
    assert isinstance(built, Ok)
    env = built.value
    eph = EphemerisConfig()
    time = EnvTime(0.0, eph.epoch_utc_s)
    first = env.evaluate(time, _shutter(), np.random.default_rng(3))
    second = env.evaluate(time, _shutter(), np.random.default_rng(99), first.truth.plume)
    assert first.truth.plume.cog_ecef_m is not None
    assert second.truth.plume.cog_ecef_m == first.truth.plume.cog_ecef_m


def test_mean_along_track_gap_matches_intensity() -> None:
    """Independent first draws have mean along-track gap near 1/lambda."""
    camera = camera_from_sensor(SensorConfig())
    dens = 2.0e-3
    half_km = 5.0
    cfg = EnvironmentConfig(
        plume="poisson_latitude",
        poisson_latitude=PoissonLatitudeParams(
            signed_lat_deg=(-90.0, 90.0),
            dens_per_km2=(dens, dens),
            cross_track_half_km=half_km,
        ),
    )
    built = build_environment(cfg, camera)
    assert isinstance(built, Ok)
    env = built.value
    eph = EphemerisConfig()
    time = EnvTime(0.0, eph.epoch_utc_s)
    gaps: list[float] = []
    for seed in range(400):
        sample = env.evaluate(time, _shutter(), np.random.default_rng(seed))
        cog = sample.truth.plume.cog_ecef_m
        assert sample.truth.plume.present is True
        assert cog is not None
        ahead_m = _along_track_ahead_m(
            sample.truth.iss, cog, eph.omega_earth_rad_s, eph.epoch_utc_s
        )
        gaps.append(ahead_m / 1000.0)
    lam = along_track_intensity_per_km(dens, half_km)
    mean = float(np.mean(np.asarray(gaps, dtype=np.float64)))
    expected = 1.0 / lam
    assert abs(mean - expected) / expected < 0.20


def test_ecef_column_ignores_rng() -> None:
    """Frozen ECEF column CoG does not depend on the generator."""
    camera = camera_from_sensor(SensorConfig())
    cfg = EnvironmentConfig(plume="ecef_column")
    built = build_environment(cfg, camera)
    assert isinstance(built, Ok)
    env = built.value
    eph = EphemerisConfig()
    time = EnvTime(0.0, eph.epoch_utc_s)
    a = env.evaluate(time, _shutter(), np.random.default_rng(1))
    b = env.evaluate(time, _shutter(), np.random.default_rng(2))
    assert a.truth.plume.present is True
    assert a.truth.plume.cog_ecef_m == b.truth.plume.cog_ecef_m


def test_large_along_track_gap_preserves_ground_distance() -> None:
    """Ground-track arc distance matches the requested gap, not R * atan(along/R)."""
    eph = EphemerisConfig()
    earth = Wgs84Ellipsoid(a_m=eph.wgs84_a_m, f=eph.wgs84_f)
    orbit = CircularKepler.from_ephemeris_config(eph)
    iss = orbit.state_eci(eph.epoch_utc_s)
    height_m = 2000.0
    ssp0 = _nadir_hit_ecef(earth, iss, height_m, eph.omega_earth_rad_s, eph.epoch_utc_s)
    assert ssp0 is not None
    along_m = 2_000_000.0
    cross_m = 0.0
    cog = _place_along_track(
        earth,
        orbit,
        iss,
        along_m,
        cross_m,
        height_m,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    assert cog is not None
    iss_ahead = _advance_iss_along_ground_track(
        earth,
        orbit,
        iss,
        along_m,
        height_m,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    assert iss_ahead is not None
    ssp1 = _nadir_hit_ecef(
        earth, iss_ahead, height_m, eph.omega_earth_rad_s, eph.epoch_utc_s
    )
    assert ssp1 is not None
    arc_m = _ssp_ground_separation_m(
        np.asarray(ssp0, dtype=np.float64), np.asarray(ssp1, dtype=np.float64)
    )
    assert abs(arc_m - along_m) / along_m < 0.01
    r0 = float(np.linalg.norm(np.asarray(ssp0, dtype=np.float64)))
    old_sphere_arc = r0 * math.atan(along_m / r0)
    assert abs(arc_m - old_sphere_arc) / along_m > 0.01


def test_wgs84_cog_on_height_proxy_after_latitude_gap() -> None:
    """After a long along-track step, CoG lies on the height-proxy ellipsoid."""
    eph = EphemerisConfig()
    earth = Wgs84Ellipsoid(a_m=eph.wgs84_a_m, f=eph.wgs84_f)
    orbit = CircularKepler.from_ephemeris_config(eph)
    iss = orbit.state_eci(eph.epoch_utc_s)
    height_m = 2000.0
    ssp0 = _nadir_hit_ecef(earth, iss, height_m, eph.omega_earth_rad_s, eph.epoch_utc_s)
    assert ssp0 is not None
    along_m = 800_000.0
    cog = _place_along_track(
        earth,
        orbit,
        iss,
        along_m,
        2500.0,
        height_m,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    assert cog is not None
    r0 = float(np.linalg.norm(np.asarray(ssp0, dtype=np.float64)))
    r_cog = float(np.linalg.norm(np.asarray(cog, dtype=np.float64)))
    assert abs(r_cog - r0) > 100.0
    a_h, b_h = height_proxy_semiaxes(eph.wgs84_a_m, eph.wgs84_f, height_m)
    x, y, z = cog
    ellipsoid_norm = (x * x + y * y) / (a_h * a_h) + (z * z) / (b_h * b_h)
    assert abs(ellipsoid_norm - 1.0) < 1.0e-6
    iss_ahead = _advance_iss_along_ground_track(
        earth,
        orbit,
        iss,
        along_m,
        height_m,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    assert iss_ahead is not None
    r_iss_ecef = ecef_from_eci(
        np.asarray(iss_ahead.r_m, dtype=np.float64),
        eph.omega_earth_rad_s,
        iss_ahead.epoch_utc_s,
        eph.epoch_utc_s,
    )
    look = np.asarray(cog, dtype=np.float64) - r_iss_ecef
    hit = earth.intersect_at_height(r_iss_ecef, look / float(np.linalg.norm(look)), height_m)
    assert hit is not None
    point, _slant = hit
    assert np.linalg.norm(point - np.asarray(cog, dtype=np.float64)) < 1.0


def test_poisson_latitude_rejects_unsorted_lats() -> None:
    """build_environment returns Err when signed_lat_deg is not increasing."""
    camera = camera_from_sensor(SensorConfig())
    cfg = EnvironmentConfig(
        plume="poisson_latitude",
        poisson_latitude=PoissonLatitudeParams(
            signed_lat_deg=(10.0, -10.0),
            dens_per_km2=(1.0e-3, 1.0e-3),
        ),
    )
    built = build_environment(cfg, camera)
    assert isinstance(built, Err)
