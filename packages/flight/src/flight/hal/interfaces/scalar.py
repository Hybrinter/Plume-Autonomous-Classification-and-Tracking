"""Scalar housekeeping-sensor hardware abstraction.

Defines the ScalarSensor protocol: a single float reading (e.g. a temperature in
Celsius or a power draw in Watts) sampled from a housekeeping monitor. Shared by the
thermal and electrical subsystems; the meaning and units of the reading are owned by
the consuming subsystem, not the sensor.
"""

from typing import Protocol, runtime_checkable

from flight.libs.types import FaultCode, Result


@runtime_checkable
class ScalarSensor(Protocol):
    """Hardware abstraction for a single-value housekeeping sensor."""

    def read(self) -> Result[float, FaultCode]:
        """Sample the current scalar reading.

        Returns:
            Result[float, FaultCode]: Ok(value) on success; Err(code) on a read error.
        """
        ...
