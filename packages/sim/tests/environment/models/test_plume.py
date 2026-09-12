"""Tests for named plume models, including PoissonLatitude draws."""

from __future__ import annotations

import numpy as np
from flight.libs.config import EphemerisConfig, SensorConfig
from flight.libs.types import Err, Ok
from sim.environment import EnvironmentConfig, build_environment, camera_from_sensor
from sim.environment.config import PoissonLatitudeParams
from sim.environment.models.plume import _along_track_ahead_m, along_track_intensity_per_km
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
