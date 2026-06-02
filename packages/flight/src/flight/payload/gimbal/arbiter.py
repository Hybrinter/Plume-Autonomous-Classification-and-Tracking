"""
Gimbal arbiter state machine for PACT.

Implements the four-state + SAFE arbiter that governs all gimbal commands.
The arbiter is a pure function: GimbalArbiter.step() has no side effects and holds
no references to queues, hardware, or I/O. All state transitions are returned as
TelemetryEventMsg values to be dispatched by the caller (process.py).

Satisfies: REQ-AIML-GIMB-001 through 008, REQ-GIMB-HIGH-001 through 004
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from flight.libs.config import ControllerConfig
from flight.libs.messages import (
    BlobMeta,
    GimbalCommandMsg,
    InferenceResultMsg,
    TelemetryEventMsg,
    utc_now_iso,
)
from flight.libs.types import GimbalState, MessageType

# Approximate pixel-to-degree conversion factor for the PACT gimbal FOV.
# 1 pixel ~= 0.04 degrees at nominal orbital altitude (~420 km).
PIXEL_TO_DEG: float = 0.04


@dataclass(frozen=True)
class ArbiterState:
    """Immutable arbiter state snapshot.

    Mirrors a Rust struct -- all fields are value types or immutable collections.
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
    current_target_id: int | None
    scan_pan_deg: float = 0.0  # current pan position during SCAN mode


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
        IDLE       -> ACQUIRING  : blob detected, confidence gate passed, persistence < threshold
        ACQUIRING  -> TRACKING   : persistence >= acquire_persistence_frames
        TRACKING   -> IDLE       : all blobs lost (or target blob disappears)
        IDLE       -> SCAN       : idle_duration_s > config.scan_entry_idle_seconds
        SCAN       -> ACQUIRING  : blob detected above threshold
        any        -> SAFE       : fault signal present in InferenceResultMsg.mode_flags

    Design Notes
    ------------
    - step() is a **pure function**: no I/O, no queue access, no random, no time.time() calls.
      The caller supplies `now` (Unix timestamp) so that the function is fully deterministic
      and trivially testable without mocking.
    - GimbalArbiter itself is stateless -- it stores no mutable instance state. ArbiterState
      is threaded externally through the process loop in process.py.
    - The ACQUIRING -> TRACKING transition uses persistence_count from BlobMeta, which is
      maintained by tracker.match_blobs(). The arbiter itself does not count frames.
    - Safety gates (confidence, min area, deadband, rate limit) are applied by process.py
      BEFORE step() is called. By the time step() sees blobs, they have passed all gates.
    """

    def __init__(self, cfg: ControllerConfig) -> None:
        self._cfg = cfg

    def step(
        self,
        state: ArbiterState,
        result: InferenceResultMsg,
        now: float,
    ) -> tuple[
        ArbiterState,
        GimbalCommandMsg | None,
        list[TelemetryEventMsg],
    ]:
        """Advance the state machine by one frame.

        Parameters
        ----------
        state:
            Current immutable arbiter state.
        result:
            Pre-filtered InferenceResultMsg (blobs already passed
            all safety gates).
        now:
            Current Unix timestamp in seconds (supplied by caller
            for determinism).

        Returns
        -------
        (new_state, command, telemetry_events)
            new_state      : Updated ArbiterState after this step.
            command         : GimbalCommandMsg to issue, or None.
            telemetry_events : Zero or more TelemetryEventMsg for
                               state transitions.
        """
        cfg = self._cfg
        old_gs = state.gimbal_state
        blobs = result.blobs
        has_blobs = len(blobs) > 0
        events: list[TelemetryEventMsg] = []
        command: GimbalCommandMsg | None = None

        new_gs = old_gs
        idle_dur = state.idle_duration_s
        target_id = state.current_target_id
        scan_pan = state.scan_pan_deg
        last_cmd_time = state.last_command_time

        # Any non-zero mode_flags signals a fault -- enter SAFE unconditionally.
        if result.mode_flags != 0 and old_gs != GimbalState.SAFE:
            new_state = replace(
                state,
                gimbal_state=GimbalState.SAFE,
                tracked_blobs=blobs,
            )
            events.append(TelemetryEventMsg(
                msg_type=MessageType.TELEMETRY_EVENT,
                timestamp_utc=utc_now_iso(),
                subsystem="controller",
                event_name="state_transition",
                payload={"from": old_gs.value, "to": GimbalState.SAFE.value},
            ))
            return new_state, None, events

        if old_gs == GimbalState.SAFE:
            # SAFE exits only via mode_queue (not handled here)
            new_gs = GimbalState.SAFE

        elif old_gs == GimbalState.IDLE:
            if has_blobs:
                if _any_acquired(blobs, cfg):
                    new_gs = GimbalState.TRACKING
                else:
                    new_gs = GimbalState.ACQUIRING
                idle_dur = 0.0
            else:
                idle_dur = state.idle_duration_s + cfg.kalman_dt_s
                if idle_dur >= cfg.scan_entry_idle_seconds:
                    new_gs = GimbalState.SCAN
                    scan_pan = 0.0

        elif old_gs == GimbalState.ACQUIRING:
            if not has_blobs:
                new_gs = GimbalState.IDLE
                idle_dur = 0.0
                target_id = None
            elif _any_acquired(blobs, cfg):
                new_gs = GimbalState.TRACKING

        elif old_gs == GimbalState.TRACKING:
            if not has_blobs:
                new_gs = GimbalState.IDLE
                idle_dur = 0.0
                target_id = None

        elif old_gs == GimbalState.SCAN:
            if has_blobs:
                if _any_acquired(blobs, cfg):
                    new_gs = GimbalState.TRACKING
                else:
                    new_gs = GimbalState.ACQUIRING
                idle_dur = 0.0

        # Emit telemetry on every state transition
        if new_gs != old_gs:
            events.append(TelemetryEventMsg(
                msg_type=MessageType.TELEMETRY_EVENT,
                timestamp_utc=utc_now_iso(),
                subsystem="controller",
                event_name="state_transition",
                payload={
                    "from": old_gs.value,
                    "to": new_gs.value,
                },
            ))

        # Generate commands based on new state
        if new_gs == GimbalState.TRACKING and has_blobs:
            best = _select_best_target(blobs)
            target_id = best.blob_id
            az_delta = best.centroid_raw[0] * PIXEL_TO_DEG
            el_delta = best.centroid_raw[1] * PIXEL_TO_DEG

            if _rate_ok(last_cmd_time, now, cfg.retarget_rate_limit_hz):
                command = GimbalCommandMsg(
                    msg_type=MessageType.GIMBAL_COMMAND,
                    timestamp_utc=utc_now_iso(),
                    frame_id=result.frame_id,
                    az_delta_deg=az_delta,
                    el_delta_deg=el_delta,
                    state=new_gs,
                    reason="tracking_target",
                )
                last_cmd_time = now

        elif new_gs == GimbalState.SCAN:
            if _rate_ok(last_cmd_time, now, cfg.retarget_rate_limit_hz):
                scan_step = (
                    cfg.scan_slew_rate_deg_per_s
                    * (1.0 / cfg.retarget_rate_limit_hz)
                )
                scan_pan = scan_pan + scan_step
                if scan_pan > 30.0:
                    scan_pan = 30.0
                    scan_step = 0.0
                elif scan_pan < -30.0:
                    scan_pan = -30.0
                    scan_step = 0.0

                command = GimbalCommandMsg(
                    msg_type=MessageType.GIMBAL_COMMAND,
                    timestamp_utc=utc_now_iso(),
                    frame_id=result.frame_id,
                    az_delta_deg=scan_step,
                    el_delta_deg=0.0,
                    state=new_gs,
                    reason="nadir_scan",
                )
                last_cmd_time = now

        new_state = ArbiterState(
            gimbal_state=new_gs,
            tracked_blobs=blobs,
            idle_duration_s=idle_dur,
            last_command_time=last_cmd_time,
            current_target_id=target_id,
            scan_pan_deg=scan_pan,
        )
        return new_state, command, events


def _any_acquired(
    blobs: tuple[BlobMeta, ...],
    cfg: ControllerConfig,
) -> bool:
    """Return True if any blob has persistence >= acquire threshold."""
    return any(
        b.persistence_count >= cfg.acquire_persistence_frames
        for b in blobs
    )


def _select_best_target(blobs: tuple[BlobMeta, ...]) -> BlobMeta:
    """Select best target: highest persistence, then confidence."""
    return max(
        blobs,
        key=lambda b: (b.persistence_count, b.mean_confidence),
    )


def _rate_ok(
    last_cmd_time: float, now: float, rate_hz: float,
) -> bool:
    """Check if enough time elapsed for a new command."""
    if rate_hz <= 0.0:
        return False
    return (now - last_cmd_time) >= (1.0 / rate_hz)
