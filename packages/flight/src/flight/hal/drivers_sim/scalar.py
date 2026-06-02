"""Simulated scalar housekeeping sensor.

Replays a fixed list of readings in order, holding the final value once exhausted (a
real housekeeping sensor always reads something). Satisfies ScalarSensor structurally.
"""

from flight.libs.types import FaultCode, Ok, Result


class SimScalarSensor:
    """Scalar sensor that replays scripted readings, holding the last (sim/SIL driver)."""

    def __init__(self, readings: list[float]) -> None:
        """Initialize with the readings to replay, in order.

        Args:
            readings: Non-empty list of readings; read() holds the last once exhausted.
        """
        self._readings = readings
        self._index = 0

    def read(self) -> Result[float, FaultCode]:
        """Return the next reading, holding the final value once exhausted."""
        index = min(self._index, len(self._readings) - 1)
        self._index += 1
        return Ok(self._readings[index])
