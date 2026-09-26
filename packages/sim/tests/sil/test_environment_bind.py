"""SIL environment bind: opt-in evaluate after catch-up, before acquire."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from flight.core.select_drivers import SimDriverInputs
from flight.hal.drivers_sim import SimGimbal, SimSensor
from flight.libs.config import DriverConfig, EphemerisConfig, PactConfig, SensorConfig
from flight.libs.time import ManualClock
from flight.libs.types import GimbalState, Ok
from sim.environment import (
    Environment,
    EnvironmentConfig,
    build_environment,
    camera_from_sensor,
)
from sim.environment.config import EcefColumnParams
from sim.environment.records import DriverFeed, EnvSample, EnvTime, PlumeState, ShutterPose
from sim.scene import build_frames, plume_detector
from sim.sil import (
    SilEnvironmentBind,
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


def _inject_appearance_mosaic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force evaluate to emit a dummy mosaic so mix guards can be exercised."""
    orig = Environment.evaluate

    def evaluate(
        self: Environment,
        time: EnvTime,
        shutter: ShutterPose,
        rng: np.random.Generator,
        prior_plume: PlumeState | None = None,
    ) -> EnvSample:
        sample = orig(self, time, shutter, rng, prior_plume)
        mosaic = np.zeros((8, 8), dtype=np.uint16)  # np.ndarray[uint16, (8, 8)]
        feed = DriverFeed(mosaic=mosaic, mask=sample.feed.mask)
        return EnvSample(truth=sample.truth, feed=feed)

    monkeypatch.setattr(Environment, "evaluate", evaluate)


def _inject_live_mosaic(monkeypatch: pytest.MonkeyPatch, mosaic: np.ndarray) -> None:
    """Force evaluate to emit a full-size mosaic so empty-frame binds can acquire."""
    orig = Environment.evaluate

    def evaluate(
        self: Environment,
        time: EnvTime,
        shutter: ShutterPose,
        rng: np.random.Generator,
        prior_plume: PlumeState | None = None,
    ) -> EnvSample:
        sample = orig(self, time, shutter, rng, prior_plume)
        feed = DriverFeed(mosaic=mosaic, mask=sample.feed.mask)
        return EnvSample(truth=sample.truth, feed=feed)

    monkeypatch.setattr(Environment, "evaluate", evaluate)


def _frozen_nadir_column(sensor_cfg: SensorConfig) -> EnvironmentConfig:
    """Return ecef_column with the epoch-nadir CoG frozen in ECEF."""
    camera = camera_from_sensor(sensor_cfg)
    eph = EphemerisConfig()
    built = build_environment(EnvironmentConfig(plume="ecef_column"), camera, eph)
    assert isinstance(built, Ok)
    sample = built.value.evaluate(
        EnvTime(0.0, eph.epoch_utc_s),
        ShutterPose(0.0, 0.0, 13.0, 0.0),
        np.random.default_rng(0),
    )
    cog = sample.truth.plume.cog_ecef_m
    assert cog is not None
    return EnvironmentConfig(
        plume="ecef_column",
        ecef_column=EcefColumnParams(cog_ecef_m=cog),
    )


def _with_outer_dt(config: PactConfig, outer_dt_s: float) -> PactConfig:
    """Return config with controller.outer.dt_s replaced when it differs."""
    if outer_dt_s == config.controller.outer.dt_s:
        return config
    return dataclasses.replace(
        config,
        controller=dataclasses.replace(
            config.controller,
            outer=dataclasses.replace(config.controller.outer, dt_s=outer_dt_s),
        ),
    )


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


def test_bind_rejects_constructor_frames_with_appearance_mosaic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mosaic-emitting evaluate plus leftover scripted frames is rejected."""
    _inject_appearance_mosaic(monkeypatch)
    config = PactConfig()
    clock = ManualClock()
    frames = build_frames(2)
    camera = camera_from_sensor(config.sensor)
    built = build_environment(EnvironmentConfig(plume="ecef_column"), camera)
    assert isinstance(built, Ok)
    system = build_sil_system(
        config,
        clock,
        frames,
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    with pytest.raises(ValueError, match="cannot be mixed"):
        bind_sil_environment(
            built.value,
            system.sensor,
            system.gimbal,
            clock,
            config.sensor,
            frames=frames,
        )


def test_pre_step_rejects_live_mosaic_over_unread_scripted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bypassing the constructor check still refuses interleaved frame_id sources."""
    _inject_appearance_mosaic(monkeypatch)
    config = PactConfig()
    clock = ManualClock()
    frames = build_frames(2)
    camera = camera_from_sensor(config.sensor)
    built = build_environment(EnvironmentConfig(plume="ecef_column"), camera)
    assert isinstance(built, Ok)
    system = build_sil_system(
        config,
        clock,
        frames,
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    bind = SilEnvironmentBind(
        built.value,
        system.sensor,
        system.gimbal,
        clock,
        config.sensor,
    )
    with pytest.raises(ValueError, match="cannot interleave"):
        bind.pre_step(1.0)


def test_pre_step_does_not_shift_encoder_noise() -> None:
    """OracleMask bind advances the plant without consuming an encoder sample."""
    config = PactConfig()
    clock_bound = ManualClock()
    clock_plain = ManualClock()
    frames_bound = build_frames(2)
    frames_plain = build_frames(2)
    camera = camera_from_sensor(config.sensor)
    built = build_environment(EnvironmentConfig(), camera)
    assert isinstance(built, Ok)
    bound = build_sil_system(
        config,
        clock_bound,
        frames_bound,
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    plain = build_sil_system(
        config,
        clock_plain,
        frames_plain,
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    bind = bind_sil_environment(
        built.value,
        bound.sensor,
        bound.gimbal,
        clock_bound,
        config.sensor,
        frames=frames_bound,
    )
    clock_bound.advance(1.0)
    clock_plain.advance(1.0)
    bind.pre_step(1.0)
    pos_bound = bound.gimbal.read_position()
    pos_plain = plain.gimbal.read_position()
    assert isinstance(pos_bound, Ok)
    assert isinstance(pos_plain, Ok)
    assert pos_bound.value.el_deg == pos_plain.value.el_deg


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
    # The finer-pixel smear cap can be smaller than the scene rate, so elevation
    # need not rise on every step. Tracking still moves the gimbal.
    assert max(elevations) > min(elevations)
    assert elevations[-1] > 0.0
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


def test_pre_step_skips_live_mask_while_constructor_frames_remain() -> None:
    """A live mask at now=0.02 is not loaded onto a constructor frame stamped 1.0."""
    config = PactConfig()
    clock = ManualClock()
    frames = build_frames(1)
    detector = plume_detector()
    original_mask = np.array(detector._scripted_segmentor._prob_mask, copy=True)
    camera = camera_from_sensor(config.sensor)
    built = build_environment(_frozen_nadir_column(config.sensor), camera)
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
        detector=detector,
        ephemeris=system.apps.payload.ephemeris,
    )
    sample = bind.pre_step(0.02)
    assert sample.feed.mask is not None
    assert system.sensor.unread_scripted_count() == 1
    assert np.array_equal(detector._scripted_segmentor._prob_mask, original_mask)
    acquired = system.sensor.acquire_frame()
    assert isinstance(acquired, Ok)
    assert acquired.value.timestamp_s == 1.0


@pytest.mark.parametrize(
    ("step_dt", "outer_dt_s", "clock0"),
    [
        (0.1, 0.020, 0.0),
        (0.2, 0.020, 0.0),
        (0.1, 0.100, 3.0),
        (0.2, 0.050, 2.0),
    ],
)
def test_ecef_column_predictor_engages_with_aligned_shutter(
    monkeypatch: pytest.MonkeyPatch,
    step_dt: float,
    outer_dt_s: float,
    clock0: float,
) -> None:
    """Catch-up before bind aligns shutter, encoder, and frame time for the predictor."""
    live = build_frames(1)[0].planes
    assert isinstance(live, np.ndarray)
    _inject_live_mosaic(monkeypatch, live)
    config = _with_outer_dt(PactConfig(), outer_dt_s)
    clock = ManualClock(monotonic_s=clock0)
    detector = plume_detector()
    camera = camera_from_sensor(config.sensor)
    built = build_environment(_frozen_nadir_column(config.sensor), camera)
    assert isinstance(built, Ok)
    system = build_sil_system(
        config,
        clock,
        [],
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
        frames=[],
        detector=detector,
        ephemeris=system.apps.payload.ephemeris,
    )
    harness = SilHarness(system, bind=bind)
    now = clock0
    for _ in range(4):
        now += step_dt
        harness.step(now)
        assert bind.last_sample is not None
        assert bind.last_sample.truth.time.monotonic_s == pytest.approx(now)
        queue = system.apps.payload.vision_queue
        assert queue
        vision = queue[-1]
        assert vision.t_s == pytest.approx(now)
        assert vision.theta_g_rad is not None
        assert any(
            abs(sample.t_s - now) <= 1.0e-9 for sample in system.apps.payload.encoder_stream.samples
        )
        assert system.apps.payload._encoder_angle_at(now) is not None
        clock.advance(step_dt)
    state = harness._payload_state
    assert state.arbiter.gimbal_state is GimbalState.TRACKING
    assert state.target.r_cog_ecef_m is not None
    assert abs(state.target.last_omega_t_nom) > 1.0e-6
    assert abs(state.target.last_omega_scene_el) > 1.0e-6


def test_missing_encoder_bracket_leaves_theta_g_none() -> None:
    """A shutter time with no encoder bracket stays rejected, not a false prediction."""
    config = PactConfig()
    clock = ManualClock()
    detector = plume_detector()
    system = build_sil_system(
        config,
        clock,
        [],
        detector,
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    harness = SilHarness(system)
    harness.step(1.0)
    samples = system.apps.payload.encoder_stream.samples
    assert samples
    assert max(sample.t_s for sample in samples) < 10.0
    assert system.apps.payload._encoder_angle_at(10.0) is None
    raw = dataclasses.replace(build_frames(1)[0], timestamp_s=10.0)
    state, _ = system.apps.payload.process_frame(raw, harness._payload_state, 10.0)
    vision = system.apps.payload.vision_queue[-1]
    assert vision.t_s == 10.0
    assert vision.theta_g_rad is None
    assert vision.p_cog is not None
    assert state.target.r_cog_ecef_m is None
    state, _ = system.apps.payload.advance_outer(state, 1.0)
    assert state.target.r_cog_ecef_m is None
    assert state.target.last_omega_t_nom == 0.0
