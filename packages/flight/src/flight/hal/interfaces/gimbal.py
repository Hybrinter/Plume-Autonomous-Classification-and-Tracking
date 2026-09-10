"""Single-axis elevation gimbal hardware abstraction.

Image processing remains two-dimensional, but the physical actuator exposes only
one elevation axis. Concrete Xeryon and simulation drivers implement this contract.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# internal
from flight.libs.types import FaultCode, Result


@dataclass(frozen=True, slots=True)
class GimbalAxisState:
    """Timestamped elevation readback and decoded controller status."""

    position_deg: float
    velocity_deg_per_s: float | None = None
    target_position_deg: float = 0.0
    sample_timestamp_s: float = 0.0
    controller_timestamp_s: float | None = None
    motor_on: bool = False
    closed_loop: bool = False
    encoder_valid: bool = False
    at_index: bool = False
    position_reached: bool = False
    scanning: bool = False
    thermal_fault: bool = False
    encoder_fault: bool = False
    end_limit_fault: bool = False
    safety_timeout_fault: bool = False
    position_failure_fault: bool = False
    feedback_stale: bool = False
    rate_lease_expired: bool = False

    @property
    def timestamp_s(self) -> float:
        return self.sample_timestamp_s

    @property
    def controller_timestamp(self) -> float | None:
        """Alias retaining the concise controller-clock spelling."""
        return self.controller_timestamp_s


@runtime_checkable
class GimbalActuator(Protocol):
    """Non-blocking, one-axis elevation actuator contract."""

    def initialize(self) -> Result[None, FaultCode]: ...
    def shutdown(self) -> Result[None, FaultCode]: ...
    def find_index(self) -> Result[None, FaultCode]: ...
    def set_position(self, position_deg: float) -> Result[None, FaultCode]: ...
    def set_velocity(self, velocity_deg_per_s: float) -> Result[None, FaultCode]: ...
    def stop(self) -> Result[None, FaultCode]: ...
    def home(self) -> Result[None, FaultCode]: ...
    def stow(self) -> Result[None, FaultCode]: ...
    def reset_faults(self) -> Result[None, FaultCode]: ...
    def read_state(self) -> Result[GimbalAxisState, FaultCode]: ...
