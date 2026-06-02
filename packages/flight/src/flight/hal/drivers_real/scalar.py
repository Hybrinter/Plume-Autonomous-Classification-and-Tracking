"""Real scalar housekeeping sensor driver (stub).

Returns a safe nominal reading (0.0) until the flight housekeeping bus interface is
wired. Satisfies ScalarSensor; tests and CI use SimScalarSensor.
"""

from flight.libs.types import FaultCode, Ok, Result


class RealScalarSensor:
    """Housekeeping scalar sensor driver (stub). Satisfies ScalarSensor; reads 0.0."""

    def read(self) -> Result[float, FaultCode]:
        """Return a nominal 0.0 reading (stub pending hardware integration)."""
        return Ok(0.0)
