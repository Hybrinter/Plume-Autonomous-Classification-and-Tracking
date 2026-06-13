"""Conformance tests: sim/real drivers satisfy the HAL protocols."""

import socket
import sys
import types

import pytest
from flight.hal.drivers_real import RealGimbal, RealStationLink
from flight.hal.drivers_sim import SimGimbal, SimSensor, SimStationLink
from flight.hal.interfaces import GimbalActuator, ImagingSensor, StationLink
from flight.libs.config import GimbalConfig, LinkConfig
from flight.libs.time import ManualClock


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_sim_sensor_satisfies_imaging_sensor() -> None:
    """SimSensor conforms to ImagingSensor (type-checked assignment + runtime check)."""
    sensor: ImagingSensor = SimSensor([])
    assert isinstance(sensor, ImagingSensor)


def test_sim_gimbal_satisfies_gimbal_actuator() -> None:
    """SimGimbal conforms to GimbalActuator."""
    gimbal: GimbalActuator = SimGimbal(clock=ManualClock())
    assert isinstance(gimbal, GimbalActuator)


def test_sim_station_link_satisfies_station_link() -> None:
    """SimStationLink conforms to StationLink (no required args)."""
    link: StationLink = SimStationLink()
    assert isinstance(link, StationLink)


def test_real_station_link_satisfies_station_link() -> None:
    """RealStationLink conforms to StationLink (constructed with cfg + clock, then closed)."""
    cfg = LinkConfig(
        command_tcp_host="127.0.0.1",
        command_tcp_port=_free_port(),
        telemetry_udp_host="127.0.0.1",
        telemetry_udp_port=_free_port(),
        socket_timeout_s=0.5,
    )
    link = RealStationLink(cfg=cfg, clock=ManualClock())
    try:
        assert isinstance(link, StationLink)
    finally:
        link.close()


def test_real_gimbal_satisfies_gimbal_actuator(monkeypatch: pytest.MonkeyPatch) -> None:
    """RealGimbal conforms to GimbalActuator (constructed over a fake serial module)."""

    class _FakePort:
        def __init__(self, port: str, baudrate: int, timeout: float) -> None:
            pass

    fake = types.ModuleType("serial")
    fake.Serial = _FakePort  # type: ignore[attr-defined]
    fake.SerialException = type("SerialException", (Exception,), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "serial", fake)

    gimbal: GimbalActuator = RealGimbal(clock=ManualClock(), cfg=GimbalConfig(serial_port="COM3"))
    assert isinstance(gimbal, GimbalActuator)
