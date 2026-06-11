"""Conformance tests: sim/real drivers satisfy the HAL protocols."""

from flight.hal.drivers_real import RealGimbal
from flight.hal.drivers_sim import SimGimbal, SimSensor
from flight.hal.interfaces import GimbalActuator, ImagingSensor
from flight.libs.time import ManualClock


def test_sim_sensor_satisfies_imaging_sensor() -> None:
    """SimSensor conforms to ImagingSensor (type-checked assignment + runtime check)."""
    sensor: ImagingSensor = SimSensor([])
    assert isinstance(sensor, ImagingSensor)


def test_sim_gimbal_satisfies_gimbal_actuator() -> None:
    """SimGimbal conforms to GimbalActuator."""
    gimbal: GimbalActuator = SimGimbal(clock=ManualClock())
    assert isinstance(gimbal, GimbalActuator)


def test_real_gimbal_satisfies_gimbal_actuator() -> None:
    """RealGimbal conforms to GimbalActuator (constructs without hardware)."""
    gimbal: GimbalActuator = RealGimbal()
    assert isinstance(gimbal, GimbalActuator)
