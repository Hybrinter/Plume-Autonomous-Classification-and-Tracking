"""Gimbal-actuator hardware abstraction.

Defines the GimbalActuator protocol that formalizes the previously-stubbed gimbal
command path, plus the GimbalPosition readback type.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from flight.libs.messages import GimbalCommandMsg
from flight.libs.types import FaultCode, Result


@dataclass(frozen=True, slots=True)
class GimbalPosition:
    """Current gimbal pointing.

    Attributes:
        az_deg: Azimuth in degrees.
        el_deg: Elevation in degrees.
    """

    az_deg: float
    el_deg: float


@runtime_checkable
class GimbalActuator(Protocol):
    """Hardware abstraction for the payload pointing gimbal."""

    def send_command(self, command: GimbalCommandMsg) -> Result[None, FaultCode]:
        """Apply an azimuth/elevation delta command to the gimbal."""
        ...

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Read the current gimbal azimuth/elevation."""
        ...
