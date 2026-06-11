"""Tests for the real (flight-hardware) HAL driver stubs."""

import importlib.util

import pytest
from flight.hal.drivers_real import RealGimbal, RealSensor
from flight.libs.time import RealClock
from flight.libs.types import Ok


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
    assert isinstance(gimbal.read_position(), Ok)


def test_real_gimbal_stub_new_surface_returns_ok() -> None:
    """RealGimbal stubs for goto_angle, set_rate, home, stow, and read_stow_switch all return Ok."""
    gimbal = RealGimbal()
    assert isinstance(gimbal.goto_angle(10.0, 5.0), Ok)
    assert isinstance(gimbal.set_rate(2.0, -1.0), Ok)
    assert isinstance(gimbal.home(), Ok)
    assert isinstance(gimbal.stow(), Ok)
    pos_result = gimbal.read_position()
    assert isinstance(pos_result, Ok)
    assert pos_result.value.timestamp_s == 0.0
    switch_result = gimbal.read_stow_switch()
    assert isinstance(switch_result, Ok)
    assert switch_result.value is False
