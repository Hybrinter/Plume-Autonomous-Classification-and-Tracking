"""Verifies env-driven driver selection: all-sim, missing sim_inputs, link=real."""

import dataclasses
import socket

import pytest
from flight.core.select_drivers import SimDriverInputs, select_drivers
from flight.hal.drivers_real import RealStationLink
from flight.hal.drivers_sim import SimGimbal, SimScalarSensor, SimSensor, SimStationLink
from flight.libs.config import PactConfig
from flight.libs.time import ManualClock
from sim.scene import build_frames, plume_detector


def _all_sim_config() -> PactConfig:
    """A PactConfig with every environment axis forced to 'sim'."""
    base = PactConfig()
    env = dataclasses.replace(
        base.environment,
        sensor="sim",
        gimbal="sim",
        compute="sim",
        link="sim",
        clock="sim",
    )
    return dataclasses.replace(base, environment=env)


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
    assert isinstance(drivers.station, SimStationLink)
    assert isinstance(drivers.thermal_sensor, SimScalarSensor)
    assert isinstance(drivers.power_sensor, SimScalarSensor)
    assert drivers.detector is inputs.detector


def test_sim_axis_without_inputs_raises() -> None:
    """A sim axis with sim_inputs=None is a programming error -> ValueError."""
    with pytest.raises(ValueError, match="sim_inputs"):
        select_drivers(_all_sim_config(), ManualClock(), None)


def test_link_real_builds_realstationlink() -> None:
    """link='real' (others sim) builds a RealStationLink bound to a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]

    base = _all_sim_config()
    env = dataclasses.replace(base.environment, link="real")
    link_cfg = dataclasses.replace(base.link, command_tcp_port=free_port)
    config = dataclasses.replace(base, environment=env, link=link_cfg)

    drivers = select_drivers(config, ManualClock(), _sim_inputs())
    try:
        assert isinstance(drivers.station, RealStationLink)
        assert isinstance(drivers.sensor, SimSensor)
    finally:
        drivers.station.close()
