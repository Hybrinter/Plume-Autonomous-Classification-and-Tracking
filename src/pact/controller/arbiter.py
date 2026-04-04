"""
Gimbal arbiter state machine for PACT.

Implements the four-state + SAFE arbiter that governs all gimbal commands.
The arbiter is a pure function: GimbalArbiter.step() has no side effects and holds
no references to queues, hardware, or I/O. All state transitions are returned as
TelemetryEventMsg values to be dispatched by the caller (process.py).

Satisfies: REQ-AIML-GIMB-001 through 008, REQ-GIMB-HIGH-001 through 004
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pact.types.enums import GimbalState
from pact.types.messages import BlobMeta, GimbalCommandMsg, InferenceResultMsg, TelemetryEventMsg


@dataclass(frozen=True)
class ArbiterState:
    """Immutable arbiter state snapshot.

    Mirrors a Rust struct — all fields are value types or immutable collections.
    Never mutate an ArbiterState; always produce a new one via dataclasses.replace().

    Fields
    ------
    gimbal_state:
        Current state in the four-state + SAFE machine.
    tracked_blobs:
        Blobs that survived all safety gates in the previous step.
    idle_duration_s:
        Seconds the arbiter has been continuously in IDLE (reset on any non-IDLE entry).
    last_command_time:
        Unix timestamp of the most recent GimbalCommandMsg issued. Used by rate limiter.
    current_target_id:
        blob_id of the blob currently being tracked, or None if not in TRACKING.
    """

    gimbal_state: GimbalState
    tracked_blobs: tuple[BlobMeta, ...]
    idle_duration_s: float
    last_command_time: float  # Unix timestamp; 0.0 if no command has been issued yet
    current_target_id: Optional[int]


class GimbalArbiter:
    """Four-state + SAFE gimbal arbiter. REQ-AIML-GIMB-008.

    State Machine
    -------------
    States
        IDLE        No blob in view, or confidence gate has not been met.
        ACQUIRING   Blob detected and above threshold, but persistence < acquire_persistence_frames.
        TRACKING    Blob has been held for >= acquire_persistence_frames consecutive frames.
        SCAN        IDLE for > config.scan_entry_idle_seconds; execute raster slew pattern.
        SAFE        Fault-induced; minimal activity. Only exit is a cleared fault signal.

    Transitions (all logged as TelemetryEventMsg)
        IDLE       → ACQUIRING  : blob detected, confidence gate passed, persistence < threshold
        ACQUIRING  → TRACKING   : persistence >= acquire_persistence_frames
        TRACKING   → IDLE       : all blobs lost (or target blob disappears)
        IDLE       → SCAN       : idle_duration_s > config.scan_entry_idle_seconds
        SCAN       → ACQUIRING  : blob detected above threshold
        any        → SAFE       : fault signal present in InferenceResultMsg.mode_flags

    Design Notes
    ------------
    - step() is a **pure function**: no I/O, no queue access, no random, no time.time() calls.
      The caller supplies `now` (Unix timestamp) so that the function is fully deterministic
      and trivially testable without mocking.
    - GimbalArbiter itself is stateless — it stores no mutable instance state. ArbiterState
      is threaded externally through the process loop in process.py.
    - The ACQUIRING → TRACKING transition uses persistence_count from BlobMeta, which is
      maintained by tracker.match_blobs(). The arbiter itself does not count frames.
    - Safety gates (confidence, min area, deadband, rate limit) are applied by process.py
      BEFORE step() is called. By the time step() sees blobs, they have passed all gates.
    """

    def step(
        self,
        state: ArbiterState,
        result: InferenceResultMsg,
        now: float,
    ) -> tuple[ArbiterState, Optional[GimbalCommandMsg], list[TelemetryEventMsg]]:
        """Advance the state machine by one frame.

        Parameters
        ----------
        state:
            Current immutable arbiter state.
        result:
            Pre-filtered InferenceResultMsg (blobs already passed all safety gates).
        now:
            Current Unix timestamp in seconds (supplied by caller for determinism).

        Returns
        -------
        (new_state, command, telemetry_events)
            new_state      : Updated ArbiterState after this step.
            command        : GimbalCommandMsg to issue, or None if no command.
            telemetry_events : Zero or more TelemetryEventMsg for state transitions.
        """
        ...
