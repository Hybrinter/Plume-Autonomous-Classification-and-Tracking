"""Gimbal-actuator hardware abstraction.

Elevation torque plant. `GimbalActuator` commands torque, reads encoder elevation,
and sets pose-loop targets (stow / home / goto_angle). There is no azimuth axis
and no rate command.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-001, REQ-GIMB-HIGH-002.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# internal
from flight.libs.types import FaultCode, Result


@dataclass(frozen=True, slots=True)
class GimbalPosition:
    """Current gimbal elevation with encoder timestamp.

    Attributes:
        el_deg: Elevation in signed off-nadir degrees (positive along-track).
        timestamp_s: Monotonic seconds at the encoder read (from the injected Clock).
    """

    el_deg: float
    timestamp_s: float


@runtime_checkable
class GimbalActuator(Protocol):
    """Hardware abstraction for the single-axis elevation gimbal.

    Tracking and the pose loop write torque. `stow` / `home` / `goto_angle` set the
    position-loop target and arm stow-switch logic. The driver enforces the hardware
    envelope on torque, rate, and travel.
    """

    def set_torque(self, tau_nm: float) -> Result[None, FaultCode]:
        """Command motor torque in N·m. The driver clips to tau_max."""
        ...

    def goto_angle(self, el_deg: float) -> Result[None, FaultCode]:
        """Set the position-loop target elevation; the driver clamps to travel limits."""
        ...

    def home(self) -> Result[None, FaultCode]:
        """Set the position-loop target to the configured home pose."""
        ...

    def stow(self) -> Result[None, FaultCode]:
        """Set the position-loop target to stow and arm the stow switch."""
        ...

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Read timestamped encoder elevation."""
        ...

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        """Read the stow switch: True when mechanically at the stow pose."""
        ...
