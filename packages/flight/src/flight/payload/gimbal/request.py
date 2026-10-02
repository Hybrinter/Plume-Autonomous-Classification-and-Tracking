"""GimbalRequest: the pure controller's typed pose-command output.

A GimbalRequest is NOT a bus message: it flows by return value from the pure control
core to the payload app shell, which maps it onto GimbalActuator HAL calls and
publishes a GimbalCommandMsg telemetry record. Tracking torque is not a request;
the inner loop writes tau. Pose primitives are ABSOLUTE / STOW / HOME.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
import math
from dataclasses import dataclass

# internal
from flight.libs.types import Err, FaultCode, GimbalCommandMode, Ok, Result


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


@dataclass(frozen=True, slots=True)
class TravelEnvelope:
    """Explicit angular travel limits carried on a control reference.

    Attributes:
        theta_min_rad: Lower elevation bound, rad.
        theta_max_rad: Upper elevation bound, rad.
        omega_max_rad_s: Maximum commanded rate magnitude, rad/s.
    """

    theta_min_rad: float
    theta_max_rad: float
    omega_max_rad_s: float


@dataclass(frozen=True, slots=True)
class RateReference:
    """Absolute elevation-rate reference with its travel envelope.

    Attributes:
        rate_rad_s: Requested absolute rate, rad/s. May exceed the envelope cap;
            execution clips.
        envelope: Travel bounds applied by the servo path.
    """

    rate_rad_s: float
    envelope: TravelEnvelope


@dataclass(frozen=True, slots=True)
class PoseReference:
    """Elevation pose hold/track target with its travel envelope.

    Attributes:
        target_rad: Target elevation, rad. Must lie inside the envelope.
        envelope: Travel bounds applied by the servo path.
    """

    target_rad: float
    envelope: TravelEnvelope


@dataclass(frozen=True, slots=True)
class StowReference:
    """Bounded move to the stow pose with its travel envelope.

    Attributes:
        target_rad: Stow elevation, rad. Must lie inside the envelope.
        envelope: Travel bounds applied by the servo path.
        timeout_s: Bounded completion window, seconds.
    """

    target_rad: float
    envelope: TravelEnvelope
    timeout_s: float


@dataclass(frozen=True, slots=True)
class InhibitReference:
    """Actuator inhibit reference: no motion, with a reason.

    Attributes:
        reason: Nonempty reason code for telemetry/logging.
    """

    reason: str


ControlReference = RateReference | PoseReference | StowReference | InhibitReference
"""Closed union of typed motion references below the payload graphs."""


def _validate_envelope(envelope: TravelEnvelope) -> Result[None, FaultCode]:
    """Check that envelope bounds are finite, ordered, and rate-cap positive."""
    if not (
        math.isfinite(envelope.theta_min_rad)
        and math.isfinite(envelope.theta_max_rad)
        and math.isfinite(envelope.omega_max_rad_s)
    ):
        return Err(FaultCode.COMMAND_INVALID)
    if envelope.theta_min_rad >= envelope.theta_max_rad:
        return Err(FaultCode.COMMAND_INVALID)
    if envelope.omega_max_rad_s <= 0.0:
        return Err(FaultCode.COMMAND_INVALID)
    return Ok(None)


def validate_reference(reference: ControlReference) -> Result[None, FaultCode]:
    """Validate a control reference without constructing or mutating it.

    Inputs:
        reference: Rate, pose, stow, or inhibit reference.

    Outputs:
        Result[None, FaultCode]: Ok(None) when valid; Err(COMMAND_INVALID)
            otherwise. A rate may exceed the envelope cap; pose and stow targets
            must lie inside the envelope inclusive; the stow timeout must be
            finite and positive; the inhibit reason must be nonempty.
    """
    if isinstance(reference, InhibitReference):
        if not reference.reason:
            return Err(FaultCode.COMMAND_INVALID)
        return Ok(None)
    envelope_check = _validate_envelope(reference.envelope)
    if isinstance(envelope_check, Err):
        return envelope_check
    if isinstance(reference, RateReference):
        if not math.isfinite(reference.rate_rad_s):
            return Err(FaultCode.COMMAND_INVALID)
        return Ok(None)
    if not math.isfinite(reference.target_rad):
        return Err(FaultCode.COMMAND_INVALID)
    if not (
        reference.envelope.theta_min_rad <= reference.target_rad <= reference.envelope.theta_max_rad
    ):
        return Err(FaultCode.COMMAND_INVALID)
    if isinstance(reference, StowReference):
        if not math.isfinite(reference.timeout_s) or reference.timeout_s <= 0.0:
            return Err(FaultCode.COMMAND_INVALID)
    return Ok(None)
