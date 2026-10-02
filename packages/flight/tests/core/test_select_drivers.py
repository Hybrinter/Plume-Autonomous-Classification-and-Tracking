"""Verifies env-driven driver selection: all-sim, missing sim_inputs, link=real."""

import dataclasses
import socket

import flight.payload.inference as inference
import pytest
from flight.core.select_drivers import SimDriverInputs, select_drivers
from flight.hal.drivers_real import RealStationLink
from flight.hal.drivers_sim import (
    SimGimbal,
    SimIssEphemeris,
    SimScalarSensor,
    SimSensor,
    SimStationLink,
)
from flight.libs.config import PactConfig
from flight.libs.time import ManualClock
from flight.payload.inference import ScriptedDetector
from sim.scene import build_frames, plume_detector


def _all_sim_config() -> PactConfig:
    """A PactConfig with every driver axis forced to 'sim'."""
    base = PactConfig()
    drivers = dataclasses.replace(
        base.drivers,
        sensor="sim",
        gimbal="sim",
        compute="sim",
        link="sim",
        clock="sim",
        ephemeris="sim",
    )
    return dataclasses.replace(base, drivers=drivers)


def _sim_inputs() -> SimDriverInputs:
    """A populated SimDriverInputs for the all-sim path."""
    return SimDriverInputs(
        frames=build_frames(2),
        detector=plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )


def test_all_sim_returns_sim_drivers_and_passed_detector() -> None:
    """All-sim selection wires every sim driver and reuses the passed detector."""
    inputs = _sim_inputs()
    drivers = select_drivers(_all_sim_config(), ManualClock(), inputs)
    assert isinstance(drivers.sensor, SimSensor)
    assert isinstance(drivers.gimbal, SimGimbal)
    assert isinstance(drivers.ephemeris, SimIssEphemeris)
    assert isinstance(drivers.station, SimStationLink)
    assert isinstance(drivers.thermal_sensor, SimScalarSensor)
    assert isinstance(drivers.power_sensor, SimScalarSensor)
    assert drivers.detector is inputs.detector
    assert isinstance(drivers.detector, ScriptedDetector)


def test_sim_axis_without_inputs_raises() -> None:
    """A sim axis with sim_inputs=None is a programming error -> ValueError."""
    with pytest.raises(ValueError, match="sim_inputs"):
        select_drivers(_all_sim_config(), ManualClock(), None)


def test_real_compute_receives_tiled_dynamic_model_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real detector is configured for dynamic tile batches and per-tile GSD."""
    config = _all_sim_config()
    config = dataclasses.replace(
        config,
        drivers=dataclasses.replace(config.drivers, compute="real"),
        inference=dataclasses.replace(config.inference, tile_rows=4, tile_cols=8),
    )
    captured: dict[str, object] = {}
    detector = _sim_inputs().detector

    def _capture_detector(**kwargs: object) -> ScriptedDetector:
        captured.update(kwargs)
        return detector

    monkeypatch.setattr(inference, "OnnxDetector", _capture_detector)
    select_drivers(config, ManualClock(), _sim_inputs())

    tile_height = config.inference.input_height_px // config.inference.tile_rows
    tile_width = config.inference.input_width_px // config.inference.tile_cols
    assert captured["grid"] == (4, 8)
    assert captured["gsd_reference_m"] == config.inference.gsd_reference_m
    assert captured["expected_input_shape"] == (None, 3, tile_height, tile_width)
    assert captured["expected_gsd_shape"] == (None, 2)
    assert captured["expected_classifier_output_shape"] == (None, 1)
    assert captured["expected_segmentor_output_shape"] == (None, 1, tile_height, tile_width)


def test_link_real_builds_realstationlink() -> None:
    """link='real' (others sim) builds a RealStationLink bound to a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]

    base = _all_sim_config()
    drivers_cfg = dataclasses.replace(base.drivers, link="real")
    link_cfg = dataclasses.replace(base.link, command_tcp_port=free_port)
    config = dataclasses.replace(base, drivers=drivers_cfg, link=link_cfg)

    drivers = select_drivers(config, ManualClock(), _sim_inputs())
    try:
        assert isinstance(drivers.station, RealStationLink)
        assert isinstance(drivers.sensor, SimSensor)
    finally:
        drivers.station.close()
