"""Real gimbal-actuator driver (flight hardware).

Stubs the full closed-loop GimbalActuator surface pending the vendor serial/CAN link
integration for the flight PTU. goto_angle, set_rate, home, and stow return Ok(None);
read_position returns the origin with timestamp_s=0.0; read_stow_switch returns
Ok(False). Full serial-PTU implementation is deferred to Task 8.

Tests and simulation use SimGimbal.

Satisfies: REQ-AIML-GIMB-001 (interface conformance only; functional pending Task 8).
"""

from __future__ import annotations

# internal
from flight.hal.interfaces.gimbal import GimbalPosition
from flight.libs.messages import GimbalCommandMsg
from flight.libs.types import FaultCode, Ok, Result


class RealGimbal:
    """Flight gimbal driver stub.

    All command methods return Ok(None); read_position returns the origin with a
    zero timestamp; read_stow_switch returns Ok(False). Full PTU serial integration
    is deferred to Task 8.
    """

    def __init__(self, port: str | None = None) -> None:
        """Open the gimbal control link (no-op stub).

        Args:
            port: Optional vendor serial/CAN port identifier (unused until Task 8).
        """
        self._port = port

    def goto_angle(self, az_deg: float, el_deg: float) -> Result[None, FaultCode]:
        """Stub: command absolute pointing. Returns Ok(None); real PTU in Task 8.

        Args:
            az_deg: Target azimuth in degrees.
            el_deg: Target elevation in degrees.

        Returns:
            Ok(None) always (stub).
        """
        return Ok(None)

    def set_rate(
        self, az_rate_deg_per_s: float, el_rate_deg_per_s: float
    ) -> Result[None, FaultCode]:
        """Stub: command axis rates. Returns Ok(None); real PTU in Task 8.

        Args:
            az_rate_deg_per_s: Azimuth rate in deg/s.
            el_rate_deg_per_s: Elevation rate in deg/s.

        Returns:
            Ok(None) always (stub).
        """
        return Ok(None)

    def home(self) -> Result[None, FaultCode]:
        """Stub: drive to home pose. Returns Ok(None); real PTU in Task 8.

        Returns:
            Ok(None) always (stub).
        """
        return Ok(None)

    def stow(self) -> Result[None, FaultCode]:
        """Stub: drive to stow pose. Returns Ok(None); real PTU in Task 8.

        Returns:
            Ok(None) always (stub).
        """
        return Ok(None)

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Stub: read encoder angles. Returns origin with timestamp_s=0.0.

        Returns:
            Ok(GimbalPosition(0.0, 0.0, 0.0)) always (stub).
        """
        return Ok(GimbalPosition(az_deg=0.0, el_deg=0.0, timestamp_s=0.0))

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        """Stub: read stow switch. Returns Ok(False); real PTU in Task 8.

        Returns:
            Ok(False) always (stub).
        """
        return Ok(False)

    def send_command(self, command: GimbalCommandMsg) -> Result[None, FaultCode]:
        """DEPRECATED legacy delta path; removed by the pointing switchover (Task 6).

        Args:
            command: Legacy GimbalCommandMsg with az/el deltas.

        Returns:
            Ok(None) always (stub).
        """
        return Ok(None)
