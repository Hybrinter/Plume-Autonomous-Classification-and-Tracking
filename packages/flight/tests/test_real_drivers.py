"""Tests for the real (flight-hardware) HAL drivers' absent-SDK guards."""

import importlib.util
import socket

import pytest
from flight.hal.drivers_real import RealGimbal, RealSensor, RealStationLink
from flight.libs.config import GimbalConfig, LinkConfig
from flight.libs.time import ManualClock, RealClock
from flight.libs.types import Ok


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.skipif(
    importlib.util.find_spec("PySpin") is not None,
    reason="PySpin is installed; the absent-SDK guard cannot be exercised",
)
def test_real_sensor_requires_pyspin_when_absent() -> None:
    """Constructing RealSensor without PySpin raises a helpful ImportError."""
    with pytest.raises(ImportError):
        RealSensor(clock=RealClock())


@pytest.mark.skipif(
    importlib.util.find_spec("serial") is not None,
    reason="pyserial is installed; the absent-SDK guard cannot be exercised",
)
def test_real_gimbal_requires_pyserial_when_absent() -> None:
    """Constructing RealGimbal without pyserial raises a helpful ImportError."""
    with pytest.raises(ImportError):
        RealGimbal(clock=RealClock(), cfg=GimbalConfig(serial_port="COM3"))


def test_real_station_link_constructs_and_closes() -> None:
    """RealStationLink binds its TCP server and can be closed cleanly (no SDK required)."""
    cfg = LinkConfig(
        command_tcp_host="127.0.0.1",
        command_tcp_port=_free_port(),
        telemetry_udp_host="127.0.0.1",
        telemetry_udp_port=_free_port(),
        socket_timeout_s=0.5,
    )
    link = RealStationLink(cfg=cfg, clock=ManualClock())
    result = link.receive_packet()
    assert isinstance(result, Ok)
    assert result.value is None
    link.close()
