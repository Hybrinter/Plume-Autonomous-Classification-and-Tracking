"""Gimbal-actuator hardware abstraction.

Defines the GimbalActuator protocol that formalizes the closed-loop gimbal command
set (absolute angle, rate, home, stow, encoder readback, stow switch), plus the
GimbalPosition readback type carrying a monotonic encoder timestamp.

send_command (delta path) is retained temporarily as a deprecated legacy method;
it is removed by the pointing switchover in Task 6.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-001, REQ-GIMB-HIGH-002.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# internal
from flight.libs.messages import GimbalCommandMsg
from flight.libs.types import FaultCode, Result


@dataclass(frozen=True, slots=True)
class GimbalPosition:
    """Current gimbal pointing with encoder timestamp.

    Attributes:
        az_deg: Azimuth in degrees (positive right of boresight).
        el_deg: Elevation in degrees (positive above boresight).
        timestamp_s: Monotonic seconds at the encoder read (from the injected Clock).
    """

    az_deg: float
    el_deg: float
    timestamp_s: float


@runtime_checkable
class GimbalActuator(Protocol):
    """Hardware abstraction for the payload pointing gimbal.

    The closed-loop surface covers absolute-angle, rate, home, and stow commands,
    plus encoder position readback and stow-switch sensing. The legacy send_command
    delta path is kept temporarily for backward compatibility during the task-6
    pointing switchover.
    """

    def goto_angle(self, az_deg: float, el_deg: float) -> Result[None, FaultCode]:
        """Command an absolute pointing; the driver clamps to travel limits."""
        ...

    def set_rate(
        self, az_rate_deg_per_s: float, el_rate_deg_per_s: float
    ) -> Result[None, FaultCode]:
        """Command axis rates; the driver clamps to the hardware slew envelope."""
        ...

    def home(self) -> Result[None, FaultCode]:
        """Drive to the configured home pose."""
        ...

    def stow(self) -> Result[None, FaultCode]:
        """Drive to the configured stow pose (the SAFE-mode mechanical safing action)."""
        ...

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Read timestamped encoder angles."""
        ...

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        """Read the stow switch: True when mechanically at the stow pose."""
        ...

    def send_command(self, command: GimbalCommandMsg) -> Result[None, FaultCode]:
        """DEPRECATED legacy delta path; removed by the pointing switchover (Task 6)."""
        ...
