"""Real gimbal-actuator driver (flight hardware).

Stub pending the vendor serial/CAN link integration for the flight gimbal unit.
Tests and simulation use SimGimbal.
"""

from flight.hal.interfaces.gimbal import GimbalPosition
from flight.libs.messages import GimbalCommandMsg
from flight.libs.types import FaultCode, Ok, Result


class RealGimbal:
    """Flight gimbal driver (stub)."""

    def __init__(self, port: str | None = None) -> None:
        """Open the gimbal control link.

        Args:
            port: Optional vendor serial/CAN port identifier.
        """
        self._port = port

    def send_command(self, command: GimbalCommandMsg) -> Result[None, FaultCode]:
        """Send an az/el delta command. Stub pending vendor-link integration."""
        return Ok(None)

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Read current pointing. Stub returns the origin pending integration."""
        return Ok(GimbalPosition(az_deg=0.0, el_deg=0.0))
