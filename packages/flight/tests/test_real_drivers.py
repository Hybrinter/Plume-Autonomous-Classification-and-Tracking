"""Tests for the real (flight-hardware) HAL drivers' absent-SDK guards."""

import importlib.util

import pytest
from flight.hal.drivers_real import RealGimbal, RealSensor
from flight.libs.config import GimbalConfig
from flight.libs.time import RealClock


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
