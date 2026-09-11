"""SIL environment bind: opt-in evaluate before step_once."""

from __future__ import annotations

import dataclasses
import math

import pytest
from flight.core.select_drivers import SimDriverInputs
from flight.hal.drivers_sim import SimGimbal, SimSensor
from flight.libs.config import DriverConfig, PactConfig
from flight.libs.time import ManualClock
from flight.libs.types import Ok
from sim.environment import EnvironmentConfig, build_environment, camera_from_sensor
from sim.scene import build_frames, plume_detector
from sim.sil import (
    SilHarness,
    ValidationHarness,
    bind_sil_environment,
    build_sil_system,
    build_validation_system,
)


def _all_sim_config() -> PactConfig:
    """Return a PactConfig whose every deployment axis is a sim stand-in."""
    sim_drivers = DriverConfig(
        sensor="sim",
        gimbal="sim",
        compute="sim",
        link="sim",
        clock="sim",
        ephemeris="sim",
        host="x86_64",
    )
    return dataclasses.replace(PactConfig(), drivers=sim_drivers)


def test_bind_rejects_empty_frames_without_mosaic() -> None:
    """OracleMask emits no mosaic, so empty SimSensor frames are rejected."""
    config = PactConfig()
    clock = ManualClock()
    camera = camera_from_sensor(config.sensor)
    built = build_environment(EnvironmentConfig(), camera)
    assert isinstance(built, Ok)
    system = build_sil_system(
        config,
        clock,
        [],
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    with pytest.raises(ValueError, match="emits mosaics"):
        bind_sil_environment(
            built.value,
            system.sensor,
            system.gimbal,
            clock,
            config.sensor,
            frames=[],
        )


def test_true_elevation_moves_ecef_projected_centroid() -> None:
    """Gimbal motion under plume tracking shifts the ECEF pinhole centroid."""
    config = PactConfig()
    clock = ManualClock()
    frames = build_frames(8)
    detector = plume_detector()
    camera = camera_from_sensor(config.sensor)
    built = build_environment(EnvironmentConfig(plume="ecef_column"), camera)
    assert isinstance(built, Ok)
    system = build_sil_system(
        config,
        clock,
        frames,
        detector,
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    bind = bind_sil_environment(
        built.value,
        system.sensor,
        system.gimbal,
        clock,
        config.sensor,
        frames=frames,
        detector=None,
        ephemeris=system.apps.payload.ephemeris,
    )
    harness = SilHarness(system, bind=bind)
    centroids: list[tuple[float, float]] = []
    elevations: list[float] = []
    now = 0.0
    for _ in range(8):
        now += 1.0
        harness.step(now)
        assert bind.last_sample is not None
        centroid = bind.last_sample.truth.centroid_band_px
        assert centroid is not None
        centroids.append(centroid)
        elevations.append(system.gimbal.true_el_deg)
        assert bind.last_hal_iss is not None
        clock.advance(1.0)
    assert elevations[-1] > elevations[0]
    first_u, first_v = centroids[0]
    last_u, last_v = centroids[-1]
    shift = math.hypot(last_u - first_u, last_v - first_v)
    assert shift > 1.0


def test_validation_harness_pre_step_records_sample() -> None:
    """ValidationHarness.step runs the same bind pre_step as SilHarness."""
    config = _all_sim_config()
    clock = ManualClock()
    frames = build_frames(2)
    detector = plume_detector()
    camera = camera_from_sensor(config.sensor)
    built = build_environment(EnvironmentConfig(plume="ecef_column"), camera)
    assert isinstance(built, Ok)
    inputs = SimDriverInputs(
        frames=frames,
        detector=detector,
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    system = build_validation_system(config, clock, inputs)
    assert isinstance(system.sensor, SimSensor)
    assert isinstance(system.gimbal, SimGimbal)
    bind = bind_sil_environment(
        built.value,
        system.sensor,
        system.gimbal,
        clock,
        config.sensor,
        frames=frames,
        detector=None,
        ephemeris=system.apps.payload.ephemeris,
    )
    ValidationHarness(system, bind=bind).step(1.0)
    assert bind.last_sample is not None
    assert bind.last_sample.truth.plume.frame == "ecef"
