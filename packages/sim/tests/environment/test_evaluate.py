"""Tests for sim.environment evaluate, orbit match, and pinhole round-trip."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from flight.hal.drivers_sim import SimIssEphemeris
from flight.libs.config import EphemerisConfig, SensorConfig
from flight.libs.time import ManualClock
from flight.libs.types import Ok
from sim.environment import EnvironmentConfig, build_environment, camera_from_sensor
from sim.environment.config_loader import load_environment_config
from sim.environment.records import EnvTime, ShutterPose


def _repo_root() -> Path:
    """Locate the repo root via packages/sim/config/environment.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "packages" / "sim" / "config" / "environment.toml").exists():
            return parent
    raise FileNotFoundError("could not find packages/sim/config/environment.toml")


def test_env_time_from_step_uses_clock_offset() -> None:
    """UTC follows payload _read_iss_at: clock.utc + (now - clock.monotonic)."""
    clock = ManualClock(monotonic_s=10.0, utc_s=1_000.0)
    time = EnvTime.from_step(clock, now=11.0)
    assert time.monotonic_s == 11.0
    assert time.utc_s == 1_001.0


def test_circular_kepler_matches_sim_iss_ephemeris() -> None:
    """Zero-error truth orbit equals the HAL sim driver at the same UTC."""
    eph = EphemerisConfig()
    clock = ManualClock(utc_s=eph.epoch_utc_s)
    driver = SimIssEphemeris(clock, eph)
    camera = camera_from_sensor(SensorConfig())
    built = build_environment(EnvironmentConfig(), camera, eph)
    assert isinstance(built, Ok)
    env = built.value
    utc = eph.epoch_utc_s + 120.0
    sample = env.evaluate(
        EnvTime(monotonic_s=120.0, utc_s=utc),
        ShutterPose(0.0, 0.0, 13.0, 0.0),
        np.random.default_rng(0),
    )
    hal = driver.read_state(utc)
    assert isinstance(hal, Ok)
    assert sample.truth.iss.r_m == hal.value.r_m
    assert sample.truth.iss.v_m_s == hal.value.v_m_s


def test_bandplane_skips_ecef_look() -> None:
    """bandplane_gaussian copies the CI centroid and does not run ECEF look."""
    camera = camera_from_sensor(SensorConfig())
    built = build_environment(EnvironmentConfig(), camera)
    assert isinstance(built, Ok)
    sample = built.value.evaluate(
        EnvTime(0.0, EphemerisConfig().epoch_utc_s),
        ShutterPose(math.radians(10.0), 0.0, 13.0, 0.0),
        np.random.default_rng(0),
    )
    assert sample.truth.plume.frame == "bandplane"
    assert sample.truth.centroid_band_px == (612.0, 124.0)
    assert sample.truth.look.el_rad == 0.0
    assert sample.truth.look.visible is False
    assert sample.feed.mosaic is None
    assert sample.feed.mask is not None


def test_nadir_ecef_column_projects_to_principal_point() -> None:
    """Nadir CoG at true_el=0 lands on the band-plane principal point."""
    eph = EphemerisConfig()
    camera = camera_from_sensor(SensorConfig())
    cfg = EnvironmentConfig(plume="ecef_column")
    built = build_environment(cfg, camera, eph)
    assert isinstance(built, Ok)
    sample = built.value.evaluate(
        EnvTime(0.0, eph.epoch_utc_s),
        ShutterPose(0.0, 0.0, 13.0, 0.0),
        np.random.default_rng(0),
    )
    assert sample.truth.plume.frame == "ecef"
    assert sample.truth.plume.cog_ecef_m is not None
    assert sample.truth.centroid_band_px is not None
    u_px, v_px = sample.truth.centroid_band_px
    assert abs(u_px - camera.width_px / 2.0) < 1e-6
    assert abs(v_px - camera.height_px / 2.0) < 1e-6
    assert sample.truth.look.visible is True


def test_pinhole_round_trip_intersect_cog() -> None:
    """project_centroid inverts intersect_cog at a small elevation."""
    from flight.payload.gimbal.intersect import intersect_cog

    eph = EphemerisConfig()
    camera = camera_from_sensor(SensorConfig())
    cfg = EnvironmentConfig(plume="ecef_column")
    built = build_environment(cfg, camera, eph)
    assert isinstance(built, Ok)
    env = built.value
    theta = math.radians(5.0)
    sample = env.evaluate(
        EnvTime(0.0, eph.epoch_utc_s),
        ShutterPose(theta, 0.0, 13.0, 0.0),
        np.random.default_rng(0),
    )
    cog = sample.truth.plume.cog_ecef_m
    centroid = sample.truth.centroid_band_px
    assert cog is not None and centroid is not None
    iss = sample.truth.iss
    hit = intersect_cog(
        p_cog_px=centroid,
        theta_g_rad=theta,
        r_iss_eci_m=iss.r_m,
        v_iss_eci_m_s=iss.v_m_s,
        utc_s=iss.epoch_utc_s,
        epoch_utc_s=eph.epoch_utc_s,
        omega_earth_rad_s=eph.omega_earth_rad_s,
        wgs84_a_m=eph.wgs84_a_m,
        wgs84_f=eph.wgs84_f,
        camera=camera,
        height_m=2000.0,
    )
    assert hit is not None
    assert abs(hit.point_ecef_m[0] - cog[0]) < 5.0
    assert abs(hit.point_ecef_m[1] - cog[1]) < 5.0
    assert abs(hit.point_ecef_m[2] - cog[2]) < 5.0


def test_far_side_ecef_column_suppresses_centroid() -> None:
    """A CoG behind Earth has no projected centroid and a zero oracle mask."""
    from sim.environment.config import EcefColumnParams

    eph = EphemerisConfig()
    camera = camera_from_sensor(SensorConfig())
    cfg = EnvironmentConfig(
        plume="ecef_column",
        ecef_column=EcefColumnParams(cog_ecef_m=(-eph.wgs84_a_m - 2000.0, 0.0, 0.0)),
    )
    built = build_environment(cfg, camera, eph)
    assert isinstance(built, Ok)
    sample = built.value.evaluate(
        EnvTime(0.0, eph.epoch_utc_s),
        ShutterPose(0.0, 0.0, 13.0, 0.0),
        np.random.default_rng(0),
    )
    assert sample.truth.look.visible is False
    assert sample.truth.centroid_band_px is None
    assert sample.feed.mask is not None
    assert float(np.max(sample.feed.mask)) == 0.0


def test_load_default_environment_toml() -> None:
    """packages/sim/config/environment.toml parses to the Python defaults."""
    path = _repo_root() / "packages" / "sim" / "config" / "environment.toml"
    result = load_environment_config(str(path))
    assert isinstance(result, Ok)
    assert result.value == EnvironmentConfig()
