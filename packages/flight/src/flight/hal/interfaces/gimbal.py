"""Gimbal-actuator hardware abstraction.

`GimbalActuator` is the detailed-plant torque surface. Production hardware also
implements `GimbalRateActuator`, which accepts signed rate commands with leases.
`stow` / `home` / `goto_angle` latch a pose target for stow-switch arming. There
is no azimuth axis.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-001, REQ-GIMB-HIGH-002.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# internal
from flight.libs.types import FaultCode, Result


@dataclass(frozen=True, slots=True)
class GimbalRateCommand:
    """Signed absolute elevation-rate command with a monotonic lease deadline.

    ``rate_deg_per_s`` is the requested physical rate.  The real Xeryon adapter
    quantizes it to the configured controller quantum before sending it.  A
    deadline is mandatory so a disconnected host cannot leave drive authority
    armed indefinitely.
    """

    rate_deg_per_s: float
    valid_until_s: float


# Descriptive alias for callers that prefer the protocol terminology.
SignedRateCommand = GimbalRateCommand


@dataclass(frozen=True, slots=True)
class GimbalPosition:
    """Current gimbal elevation with encoder timestamp.

    Attributes:
        el_deg: Elevation in signed off-nadir degrees (positive along-track).
        timestamp_s: Mapped encoder sample time in the application monotonic domain.
            This is not host receipt time.
        raw_controller_time_s: Raw controller timestamp, when supplied by the device.
        sequence: Monotonic feedback-frame sequence, or zero when unavailable.
        status_bits: Raw controller status bit field.
        time_mapping_uncertainty_s: Bound on mapping uncertainty in seconds.
    """

    el_deg: float
    timestamp_s: float
    raw_controller_time_s: float | None = None
    sequence: int = 0
    status_bits: int = 0
    time_mapping_uncertainty_s: float = 0.0


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
    controller_status_bits: int = 0
    motor_on: bool = False
    closed_loop: bool = False
    duty_credit_s: float | None = None
    duty_locked_out: bool = False
    watchdog_gate_confirmed: bool = False
    time_mapping_valid: bool = False
    requested_rate_deg_per_s: float | None = None
    quantized_rate_deg_per_s: float | None = None


@runtime_checkable
class GimbalActuator(Protocol):
    """Detailed-plant hardware abstraction for the single-axis elevation gimbal.

    Tracking and STOW / HOME / GOTO write torque only in detailed-plant SIL.
    Production rate-command drivers use :class:`GimbalRateActuator`.
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


@runtime_checkable
class ExternalWatchdogGate(Protocol):
    """Independent hardware gate that can remove physical motion authority.

    Serial acknowledgement from the XD-C is not sufficient evidence of
    inhibition.  The production driver requires this gate to acknowledge the
    inhibit before reporting containment.
    """

    def request_inhibit(self, reason: str) -> Result[None, FaultCode]:
        """Request removal of motor authority at the independent gate."""
        ...

    def inhibit_confirmed(self) -> Result[bool, FaultCode]:
        """Return physical confirmation that motion authority is removed."""
        ...


@runtime_checkable
class GimbalRateActuator(GimbalActuator, Protocol):
    """Rate-command extension implemented by the production Xeryon adapter.

    The legacy ``GimbalActuator`` torque method remains for detailed-plant SIL
    compatibility during migration; production hardware implements this rate
    interface and rejects torque commands.
    """

    def set_rate(self, command: GimbalRateCommand) -> Result[None, FaultCode]:
        """Command a signed absolute rate until ``command.valid_until_s``."""
        ...

    def stow_reference_step(self, now_s: float | None = None) -> Result[bool, FaultCode]:
        """Advance bounded switch-referenced stow; return true when inhibited at stow."""
        ...

    def shutdown(self) -> Result[None, FaultCode]:
        """Close transport only after confirmed inhibit; never home on shutdown."""
        ...
