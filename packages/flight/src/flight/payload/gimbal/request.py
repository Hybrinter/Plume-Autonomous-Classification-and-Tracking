"""GimbalRequest: the pure controller's typed pose-command output.

A GimbalRequest is NOT a bus message: it flows by return value from the pure control
core to the payload app shell, which maps it onto GimbalActuator HAL calls and
publishes a GimbalCommandMsg telemetry record. Tracking torque is not a request;
the inner loop writes tau. Pose primitives are ABSOLUTE / STOW / HOME.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# internal
from flight.libs.types import GimbalCommandMode


@dataclass(frozen=True, slots=True)
class GimbalRequest:
    """One pose command decided by the pure control core.

    Attributes:
        mode: ABSOLUTE, STOW, or HOME.
        el_deg: Target elevation in degrees for ABSOLUTE; ignored for STOW/HOME.
        reason: Human-readable reason code for telemetry/logging.

    Notes:
        GimbalRequest never travels on the bus. The inner loop turns the outer rate
        reference into torque; this type is only the pose-loop setpoint.
    """

    mode: GimbalCommandMode
    el_deg: float
    reason: str
