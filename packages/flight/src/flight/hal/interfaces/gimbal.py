"""Gimbal-actuator hardware abstraction.

Elevation torque plant. `GimbalActuator` commands torque and reads encoder
elevation. `stow` / `home` / `goto_angle` latch a pose target for stow-switch
arming. The payload position loop writes the rate reference `r` into the inner
PI; the driver does not close a position or rate loop. There is no azimuth axis
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


@dataclass(frozen=True, slots=True)
class GimbalHealth:
    """Actuator safety evidence exposed by the gimbal driver.

    ``inhibit_confirmed`` only means that the driver has evidence that its output
    stage is inhibited.  A driver which cannot independently inhibit the motor
    must return ``Err`` from :meth:`GimbalActuator.inhibit`, rather than claiming
    this condition.
    """

    feedback_valid: bool
    last_feedback_s: float | None
    command_valid_until_s: float | None
    inhibited: bool
    inhibit_confirmed: bool


@runtime_checkable
class GimbalActuator(Protocol):
    """Hardware abstraction for the single-axis elevation gimbal.

    Tracking and STOW / HOME / GOTO write torque. `stow` / `home` / `goto_angle`
    latch a pose target and arm stow-switch logic. The driver does not run an
    internal position or rate controller.
    """

    def set_torque(
        self, tau_nm: float, valid_until_s: float | None = None
    ) -> Result[None, FaultCode]:
        """Command motor torque in N·m until an absolute monotonic deadline.

        A supplied deadline is command authority, not a suggested refresh period:
        after it expires the driver must independently inhibit drive output.
        """
        ...

    def inhibit(self, reason: str) -> Result[GimbalHealth, FaultCode]:
        """Immediately remove drive authority and return confirmed inhibit evidence."""
        ...

    def read_health(self) -> Result[GimbalHealth, FaultCode]:
        """Return current feedback, command-authority, and inhibit evidence."""
        ...

    def goto_angle(self, el_deg: float) -> Result[None, FaultCode]:
        """Latch a pose target. Motion still comes from `set_torque`."""
        ...

    def home(self) -> Result[None, FaultCode]:
        """Latch the configured home pose as the pose target."""
        ...

    def stow(self) -> Result[None, FaultCode]:
        """Latch the stow pose and arm the stow switch."""
        ...

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Read timestamped encoder elevation."""
        ...

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        """Read the stow switch: True when mechanically at the stow pose."""
        ...
