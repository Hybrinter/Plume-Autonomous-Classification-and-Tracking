"""Planned capture execution: deadline schedule, duty floor, inference decimation (pure).

The capture loop is the only production/SIL acquire seam; this module owns its
timing decisions as pure functions so every caller sees identical cadence. A
due call spends exactly one opportunity and schedules the next deadline at
now + capture_interval_s, so a late call never bursts missed frames. The duty
floor selects which due opportunities acquire; the remaining due opportunities
drain the running stream. record_capture is the successful-capture seam: it
counts accepted frames per capture context and decimates inference to the
first success and then every Nth frame.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

from flight.libs.types import Err, FaultCode, Ok, Result
from flight.payload.graphs.base import (
    EffectivePolicy,
    InferencePolicy,
    PolicyLimits,
    validate_policy,
)
from flight.payload.records import CaptureContext


class CaptureDecision(Enum):
    """What the capture loop may do for one call under the plan."""

    WAIT = "wait"
    DRAIN = "drain"
    CAPTURE = "capture"


@dataclass(frozen=True, slots=True)
class CaptureSchedule:
    """Phase bookkeeping for planned capture execution.

    Attributes:
        context: Capture context the counters belong to; a changed context
            (activation, policy revision, or containment token) resets phase.
        next_opportunity_s: Deadline of the next acquire opportunity, or None
            before the first (immediately due).
        opportunities: Due opportunities spent under this context.
        captured_frames: Successful captures recorded under this context.
    """

    context: CaptureContext | None = None
    next_opportunity_s: float | None = None
    opportunities: int = 0
    captured_frames: int = 0


@dataclass(frozen=True, slots=True)
class CapturePlan:
    """One capture-loop decision plus the schedule that produced it.

    Attributes:
        schedule: Updated schedule to retain for the next call.
        decision: WAIT (nothing), DRAIN (release one buffered frame), or
            CAPTURE (acquire one frame).
    """

    schedule: CaptureSchedule
    decision: CaptureDecision


def plan_capture(
    schedule: CaptureSchedule,
    policy: EffectivePolicy,
    context: CaptureContext,
    now: float,
    limits: PolicyLimits,
) -> Result[CapturePlan, FaultCode]:
    """Decide whether this capture-loop call waits, drains, or acquires.

    Inputs:
        schedule: Phase bookkeeping from the previous call; reset when the
            passed context differs (activation, policy revision, or
            containment token changed).
        policy: Fully resolved effective policy for this context.
        context: Capture context stamped atomically by the shell.
        now: Monotonic seconds of this call; must be finite.
        limits: Validated sensor bounds used to revalidate the policy.

    Outputs:
        Result[CapturePlan, FaultCode]: Err(COMMAND_INVALID) for a nonfinite
            time or an invalid complete policy; otherwise Ok with WAIT before
            the deadline, or one spent opportunity planned as CAPTURE or
            DRAIN by the duty floor (index starts at 1; duty 0.5 captures even
            indexes). The first opportunity is immediately due at now. A
            validated policy with acquisition disabled always plans WAIT:
            nothing is spent and no deadline advances.
    """
    valid = validate_policy(policy.imaging, policy.inference, limits)
    if isinstance(valid, Err):
        return valid
    if not math.isfinite(now):
        return Err(FaultCode.COMMAND_INVALID)
    if schedule.context != context:
        schedule = CaptureSchedule(context=context)
    if not policy.imaging.acquisition_enabled:
        return Ok(CapturePlan(schedule=schedule, decision=CaptureDecision.WAIT))
    if schedule.next_opportunity_s is not None and now < schedule.next_opportunity_s:
        return Ok(CapturePlan(schedule=schedule, decision=CaptureDecision.WAIT))
    index = schedule.opportunities + 1
    duty = policy.imaging.duty_cycle
    decision = (
        CaptureDecision.CAPTURE
        if math.floor(index * duty) > math.floor((index - 1) * duty)
        else CaptureDecision.DRAIN
    )
    planned = replace(
        schedule,
        opportunities=index,
        next_opportunity_s=now + policy.imaging.capture_interval_s,
    )
    return Ok(CapturePlan(schedule=planned, decision=decision))


def record_capture(
    schedule: CaptureSchedule,
    context: CaptureContext,
    inference: InferencePolicy,
) -> tuple[CaptureSchedule, bool]:
    """Count one successful capture and decide whether inference runs on it.

    Precondition: `inference` is a fully resolved, already validated policy
    (as produced by `operating_policy` and enforced by `plan_capture`); this
    function does not revalidate it.

    Inputs:
        schedule: Phase bookkeeping; reset when the passed context differs.
        context: Capture context the successful frame belongs to.
        inference: Resolved inference policy supplying the decimation period.

    Outputs:
        tuple[CaptureSchedule, bool]: Updated schedule with captured_frames
            incremented, and True when inference runs on this frame -- the
            first successful capture under a context and then every
            every_n_frames-th capture thereafter.
    """
    if schedule.context != context:
        schedule = CaptureSchedule(context=context)
    captured = schedule.captured_frames + 1
    updated = replace(schedule, captured_frames=captured)
    return updated, inference.enabled and (captured - 1) % inference.every_n_frames == 0
