"""RealGimbal behavior tests against a fake pyserial module (no SDK in CI)."""

import sys
import types

import pytest
from flight.libs.config import GimbalConfig
from flight.libs.time import ManualClock
from flight.libs.types import Err, FaultCode, Ok


class _FakeSerial:
    """Scriptable serial port: records writes, replays queued response lines."""

    def __init__(self, port: str, baudrate: int, timeout: float) -> None:
        self.writes: list[bytes] = []
        self.responses: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def readline(self) -> bytes:
        return self.responses.pop(0) if self.responses else b""


def _install_fake_serial(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSerial]:
    """Install a scriptable fake `serial` module into sys.modules."""
    fake = types.ModuleType("serial")

    class SerialException(Exception):  # noqa: N818 -- name mirrors pyserial's real class
        pass

    fake.Serial = _FakeSerial  # type: ignore[attr-defined]
    fake.SerialException = SerialException  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "serial", fake)
    return _FakeSerial


def test_goto_angle_writes_position_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """goto_angle converts degrees to encoder counts and writes PP/TP commands."""
    _install_fake_serial(monkeypatch)
    from flight.hal.drivers_real import RealGimbal

    gimbal = RealGimbal(clock=ManualClock(), cfg=GimbalConfig(serial_port="COM3"))
    gimbal._port.responses = [b"*\n", b"*\n"]
    result = gimbal.goto_angle(10.0, -5.0)
    assert isinstance(result, Ok)
    assert gimbal._port.writes[0] == b"PP776\n"  # 10.0 deg * 77.6 counts/deg
    assert gimbal._port.writes[1] == b"TP-388\n"


def test_goto_angle_clamps_to_travel_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Targets outside the travel envelope are clamped before conversion."""
    _install_fake_serial(monkeypatch)
    from flight.hal.drivers_real import RealGimbal

    gimbal = RealGimbal(clock=ManualClock(), cfg=GimbalConfig(serial_port="COM3"))
    gimbal._port.responses = [b"*\n", b"*\n"]
    assert isinstance(gimbal.goto_angle(500.0, 0.0), Ok)
    assert gimbal._port.writes[0] == b"PP6984\n"  # clamped to az_max 90 deg


def test_error_response_is_gimbal_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    """A '!' response from the PTU maps to Err(GIMBAL_FAULT)."""
    _install_fake_serial(monkeypatch)
    from flight.hal.drivers_real import RealGimbal

    gimbal = RealGimbal(clock=ManualClock(), cfg=GimbalConfig(serial_port="COM3"))
    gimbal._port.responses = [b"! illegal command\n"]
    result = gimbal.goto_angle(1.0, 0.0)
    assert isinstance(result, Err)
    assert result.error == FaultCode.GIMBAL_FAULT


def test_read_position_parses_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """read_position queries PP/TP and converts counts back to timestamped degrees."""
    _install_fake_serial(monkeypatch)
    from flight.hal.drivers_real import RealGimbal

    clock = ManualClock()
    gimbal = RealGimbal(clock=clock, cfg=GimbalConfig(serial_port="COM3"))
    gimbal._port.responses = [b"* 776\n", b"* -388\n"]
    result = gimbal.read_position()
    assert isinstance(result, Ok)
    assert abs(result.value.az_deg - 10.0) < 1e-6
    assert abs(result.value.el_deg - (-5.0)) < 1e-6
    assert result.value.timestamp_s == clock.monotonic_s()


def test_missing_pyserial_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without pyserial installed, constructing RealGimbal raises ImportError."""
    monkeypatch.setitem(sys.modules, "serial", None)
    from flight.hal.drivers_real import RealGimbal

    with pytest.raises(ImportError):
        RealGimbal(clock=ManualClock(), cfg=GimbalConfig(serial_port="COM3"))
