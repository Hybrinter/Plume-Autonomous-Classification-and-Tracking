"""Tests for the real (flight-hardware) HAL driver stubs."""

import importlib.util

import pytest
from flight.hal.drivers_real import RealGimbal, RealSensor
from flight.libs.messages import GimbalCommandMsg
from flight.libs.time import RealClock
from flight.libs.types import GimbalState, MessageType, Ok


@pytest.mark.skipif(
    importlib.util.find_spec("PySpin") is not None,
    reason="PySpin is installed; the absent-SDK guard cannot be exercised",
)
def test_real_sensor_requires_pyspin_when_absent() -> None:
    """Constructing RealSensor without PySpin raises a helpful ImportError."""
    with pytest.raises(ImportError):
        RealSensor(clock=RealClock())


def test_real_gimbal_stub_constructs_and_reads() -> None:
    """RealGimbal stub constructs without hardware and returns a position."""
    gimbal = RealGimbal()
    command = GimbalCommandMsg(
        msg_type=MessageType.GIMBAL_COMMAND,
        timestamp_utc="2026-05-31T00:00:00.000Z",
        frame_id=1,
        az_delta_deg=1.0,
        el_delta_deg=1.0,
        state=GimbalState.TRACKING,
        reason="test",
    )
    assert isinstance(gimbal.send_command(command), Ok)
    assert isinstance(gimbal.read_position(), Ok)
