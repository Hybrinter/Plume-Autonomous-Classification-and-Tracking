"""Conformance + behavior tests for the ScalarSensor HAL and its drivers."""

from flight.hal.drivers_real import RealScalarSensor
from flight.hal.drivers_sim import SimScalarSensor
from flight.hal.interfaces import ScalarSensor
from flight.libs.types import Ok


def test_sim_scalar_sensor_satisfies_protocol() -> None:
    """SimScalarSensor conforms to ScalarSensor (typed + runtime)."""
    sensor: ScalarSensor = SimScalarSensor([1.0])
    assert isinstance(sensor, ScalarSensor)


def test_real_scalar_sensor_satisfies_protocol() -> None:
    """RealScalarSensor conforms to ScalarSensor and constructs without hardware."""
    sensor: ScalarSensor = RealScalarSensor()
    assert isinstance(sensor, ScalarSensor)


def test_sim_replays_readings_then_holds_last() -> None:
    """read() yields each scripted reading once, then holds the final value."""
    sensor = SimScalarSensor([10.0, 20.0])
    first = sensor.read()
    second = sensor.read()
    third = sensor.read()
    assert isinstance(first, Ok) and first.value == 10.0
    assert isinstance(second, Ok) and second.value == 20.0
    assert isinstance(third, Ok) and third.value == 20.0  # holds last


def test_real_scalar_sensor_reads_nominal_zero() -> None:
    """RealScalarSensor stub returns a safe nominal reading of 0.0."""
    sensor = RealScalarSensor()
    result = sensor.read()
    assert isinstance(result, Ok)
    assert result.value == 0.0
