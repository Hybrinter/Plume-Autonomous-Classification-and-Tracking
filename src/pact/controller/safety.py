"""
Safety gates for PACT controller subsystem.

All safety gate functions run BEFORE the GimbalArbiter.step() is called. They are pure
functions with no side effects. The process loop in process.py applies them in order:

    confidence gate → min area gate → (blob matching) → deadband check → rate limit → arbiter

Satisfies: REQ-AIML-DATA-008, REQ-AIML-DATA-009,
           REQ-AIML-GIMB-005, REQ-AIML-GIMB-006, REQ-AIML-GIMB-007
"""

from __future__ import annotations

from pact.types.enums import FaultCode
from pact.types.enums import Ok, Err, Result
from pact.types.messages import BlobMeta


def apply_confidence_gate(
    blobs: tuple[BlobMeta, ...],
    threshold: float,
) -> tuple[BlobMeta, ...]:
    """Reject blobs whose mean_confidence is strictly below threshold. REQ-AIML-DATA-008.

    Parameters
    ----------
    blobs:
        Candidate blobs from the current inference result.
    threshold:
        Minimum acceptable mean_confidence (exclusive lower bound).
        Sourced from ControllerConfig.confidence_gate (default 0.55).

    Returns
    -------
    tuple[BlobMeta, ...]
        Filtered blobs; may be empty.
    """
    return tuple(b for b in blobs if b.mean_confidence >= threshold)


def apply_min_area_gate(
    blobs: tuple[BlobMeta, ...],
    min_px: int,
) -> tuple[BlobMeta, ...]:
    """Reject blobs whose pixel_area is strictly below min_px. REQ-AIML-DATA-009.

    Parameters
    ----------
    blobs:
        Candidate blobs (typically already confidence-gated).
    min_px:
        Minimum acceptable pixel area (exclusive lower bound).
        Sourced from ControllerConfig.min_blob_area_px (default 15).

    Returns
    -------
    tuple[BlobMeta, ...]
        Filtered blobs; may be empty.
    """
    return tuple(b for b in blobs if b.pixel_area >= min_px)


def check_deadband(
    displacement_px: float,
    min_px: int,
    max_px: int,
) -> "Result[bool, FaultCode]":
    """Check displacement against min and max deadband thresholds.

    Interpretation
    --------------
    - Below min_px  → Ok(False): displacement is too small to warrant a command. Not a fault.
    - In [min_px, max_px] → Ok(True): displacement is actionable, issue a command.
    - Above max_px  → Err(GIMBAL_RUNAWAY): displacement exceeds safe range. This is a fault
      that must be escalated; do NOT issue a gimbal command.

    REQ-AIML-GIMB-006 (min deadband), REQ-AIML-GIMB-007 (max deadband / runaway detection).

    Parameters
    ----------
    displacement_px:
        Euclidean pixel displacement between the EMA-filtered centroid and the frame center.
    min_px:
        Minimum displacement to trigger a command (ControllerConfig.min_deadband_px, default 20).
    max_px:
        Maximum displacement before a runaway fault is raised
        (ControllerConfig.max_deadband_px, default 250).

    Returns
    -------
    Result[bool, FaultCode]
        Ok(True)  if command should be issued.
        Ok(False) if displacement is below minimum (no command, no fault).
        Err(FaultCode.GIMBAL_RUNAWAY) if displacement exceeds maximum.
    """
    if displacement_px > max_px:
        return Err(FaultCode.GIMBAL_RUNAWAY)
    if displacement_px < min_px:
        return Ok(False)
    return Ok(True)


def check_rate_limit(
    last_command_time: float,
    now: float,
    rate_limit_hz: float,
) -> bool:
    """Return True if sufficient time has elapsed since the last command. REQ-AIML-GIMB-005.

    Prevents the gimbal from being commanded faster than rate_limit_hz, which protects
    the motor drive from thermal overload and ensures commands do not stack up.

    Parameters
    ----------
    last_command_time:
        Unix timestamp of the most recent GimbalCommandMsg that was issued.
        Pass 0.0 if no command has ever been issued.
    now:
        Current Unix timestamp in seconds (supplied by caller for determinism).
    rate_limit_hz:
        Maximum command rate in Hz (ControllerConfig.retarget_rate_limit_hz, default 0.5).

    Returns
    -------
    bool
        True if a new command may be issued; False if the rate limit has not yet elapsed.
    """
    if rate_limit_hz <= 0.0:
        return False
    min_interval_s = 1.0 / rate_limit_hz
    return (now - last_command_time) >= min_interval_s
