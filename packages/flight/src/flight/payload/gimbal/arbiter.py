"""
Gimbal arbiter state machine for PACT.

Implements the four-state + SAFE arbiter that governs all gimbal commands.
The arbiter is a pure function: GimbalArbiter.step() has no side effects and holds
no references to queues, hardware, or I/O. All state transitions are returned as
TelemetryEventMsg values to be dispatched by the caller (the payload app shell).

The arbiter emits a typed GimbalRequest (not a bus message): RATE during TRACKING,
ABSOLUTE to the science limb during REWIND, STOW on SAFE entry. SAFE is latched in
the arbiter and commanded/cleared by ModeChangeMsg flags (safe_commanded/safe_cleared)
drained by the shell; pointing error is supplied as boresight-relative degrees by the
caller.

Satisfies: REQ-AIML-GIMB-001 through 008, REQ-GIMB-HIGH-001 through 004
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from flight.libs.config import ControllerConfig, GimbalConfig
from flight.libs.messages import (
    BlobMeta,
    InferenceResultMsg,
    TelemetryEventMsg,
    utc_now_iso,
)
from flight.libs.types import GimbalCommandMode, GimbalState, MessageType
from flight.payload.gimbal.request import GimbalRequest

_LIMB_ARRIVAL_DEG: float = 0.5  # treat elevation within this of science max as arrived


@dataclass(frozen=True)
class ArbiterState:
    """Immutable arbiter state snapshot.

    All fields are value types or immutable collections. Never mutate an ArbiterState;
    always produce a new one via dataclasses.replace().

    Fields
    ------
    gimbal_state:
        Current state in the four-state + SAFE machine.
    tracked_blobs:
        Blobs that survived all safety gates in the previous step.
    idle_duration_s:
        Seconds the arbiter has been continuously in IDLE (reset on any non-IDLE entry).
    last_command_time:
        Unix timestamp of the most recent command issued. Used by the rate limiter.
    current_target_id:
        blob_id of the blob currently being tracked, or None if not in TRACKING.
    miss_count:
        TRACKING release-hysteresis counter: consecutive frames with no blob while in
        TRACKING. TRACKING releases only once miss_count reaches
        release_persistence_frames; any blob resets it to 0.
    """

    gimbal_state: GimbalState
    tracked_blobs: tuple[BlobMeta, ...]
    idle_duration_s: float
    last_command_time: float  # Unix timestamp; 0.0 if no command has been issued yet
    current_target_id: int | None
    miss_count: int = 0  # TRACKING release-hysteresis counter


class GimbalArbiter:
    """Four-state + SAFE gimbal arbiter. REQ-AIML-GIMB-008.

    State Machine
    -------------
    States
        IDLE        No blob in view, or confidence gate has not been met.
        ACQUIRING   Blob detected and above threshold, but persistence < acquire frames.
        TRACKING    Blob held for >= acquire_persistence_frames consecutive frames.
        REWIND      Track lost before the science limb; slew elevation to el_science_max.
        SAFE        Fault/ground-induced; latched. Only exit is a cleared-mode signal.

    Transitions (all logged as TelemetryEventMsg)
        IDLE       -> ACQUIRING  : blob detected, persistence < acquire threshold
        ACQUIRING  -> TRACKING   : persistence >= acquire_persistence_frames
        TRACKING   -> REWIND     : no blobs for release_persistence_frames and el below limb
        TRACKING   stays TRACKING: no blobs for release_persistence_frames and el at limb
        REWIND     -> TRACKING   : arrived at science limb with no blob, or blob acquired
        REWIND     -> ACQUIRING  : blob detected above threshold before limb arrival
        any        -> SAFE       : safe_commanded or InferenceResultMsg.mode_flags != 0
        SAFE       -> IDLE       : safe_cleared (ground recovery)

    Design Notes
    ------------
    - step() is a **pure function**: no I/O, no queue access, no random, no time.time().
      The caller supplies `now` (monotonic seconds), the boresight-relative error, and
      the current elevation so the function is fully deterministic.
    - GimbalArbiter itself is stateless -- it stores no mutable instance state. ArbiterState
      is threaded externally through the app loop.
    - SAFE latches: while in SAFE, no further requests are produced and blobs are ignored
      until safe_cleared returns the machine to IDLE.
    - The TRACKING command is a proportional fallback (gain 1.0 / s) on the boresight error,
      clamped to the hardware slew cap; control.py refines it via the LQR once the estimator
      is initialized. REWIND issues ABSOLUTE elevation to the science limb.
    """

    def __init__(self, cfg: ControllerConfig, gimbal: GimbalConfig) -> None:
        """Hold the immutable controller and gimbal config for thresholds and envelopes.

        Args:
            cfg: The ControllerConfig supplying gates and persistence.
            gimbal: The GimbalConfig supplying science-limb elevation and hardware slew.
        """
        self._cfg = cfg
        self._gimbal = gimbal

    def step(
        self,
        state: ArbiterState,
        result: InferenceResultMsg,
        error_deg: tuple[float, float] | None,
        now: float,
        safe_commanded: bool,
        safe_cleared: bool,
        el_deg: float | None = None,
    ) -> tuple[
        ArbiterState,
        GimbalRequest | None,
        list[TelemetryEventMsg],
    ]:
        """Advance the state machine by one frame and emit at most one GimbalRequest.

        Parameters
        ----------
        state:
            Current immutable arbiter state.
        result:
            Pre-filtered InferenceResultMsg (blobs already passed all safety gates).
        error_deg:
            Boresight-relative (az, el) error of the matched target in degrees, or None
            when no usable target exists. Used only in TRACKING to form the rate command.
        now:
            Monotonic seconds (supplied by the caller for determinism; used as deltas).
        safe_commanded:
            True if a SAFE mode change was drained this frame: latch SAFE and stow.
        safe_cleared:
            True if a non-SAFE mode change was drained this frame: exit SAFE to IDLE.
        el_deg:
            Current elevation in signed off-nadir degrees, or None when unavailable.

        Returns
        -------
        (new_state, request, telemetry_events)
            new_state        : Updated ArbiterState after this step.
            request          : GimbalRequest to issue, or None.
            telemetry_events : Zero or more TelemetryEventMsg for state transitions.
        """
        cfg = self._cfg
        gimbal = self._gimbal
        old_gs = state.gimbal_state
        blobs = result.blobs
        has_blobs = len(blobs) > 0
        events: list[TelemetryEventMsg] = []
        at_limb = el_deg is not None and el_deg >= gimbal.el_science_max_deg - _LIMB_ARRIVAL_DEG

        # SAFE entry: a commanded SAFE or any non-zero mode_flags latches SAFE and stows.
        if (safe_commanded or result.mode_flags != 0) and old_gs != GimbalState.SAFE:
            new_state = replace(
                state,
                gimbal_state=GimbalState.SAFE,
                tracked_blobs=blobs,
                miss_count=0,
                current_target_id=None,
            )
            events.append(self._transition_event(old_gs, GimbalState.SAFE))
            stow_request = GimbalRequest(
                mode=GimbalCommandMode.STOW,
                az_deg=0.0,
                el_deg=0.0,
                reason="safe_entry_stow",
            )
            return new_state, stow_request, events

        # SAFE latch / exit: while SAFE, produce nothing unless cleared.
        if old_gs == GimbalState.SAFE:
            if safe_cleared:
                new_state = ArbiterState(
                    gimbal_state=GimbalState.IDLE,
                    tracked_blobs=(),
                    idle_duration_s=0.0,
                    last_command_time=state.last_command_time,
                    current_target_id=None,
                    miss_count=0,
                )
                events.append(self._transition_event(GimbalState.SAFE, GimbalState.IDLE))
                return new_state, None, events
            return replace(state, tracked_blobs=blobs), None, events

        new_gs = old_gs
        idle_dur = state.idle_duration_s
        target_id = state.current_target_id
        miss_count = state.miss_count
        last_cmd_time = state.last_command_time

        if old_gs == GimbalState.IDLE:
            if has_blobs:
                new_gs = (
                    GimbalState.TRACKING if _any_acquired(blobs, cfg) else GimbalState.ACQUIRING
                )
                idle_dur = 0.0
            else:
                idle_dur = state.idle_duration_s + cfg.kalman_dt_s

        elif old_gs == GimbalState.ACQUIRING:
            if not has_blobs:
                new_gs = GimbalState.IDLE
                idle_dur = 0.0
                target_id = None
            elif _any_acquired(blobs, cfg):
                new_gs = GimbalState.TRACKING

        elif old_gs == GimbalState.TRACKING:
            if has_blobs:
                miss_count = 0
            else:
                miss_count = state.miss_count + 1
                if miss_count >= cfg.release_persistence_frames:
                    if at_limb:
                        miss_count = 0
                        target_id = None
                    else:
                        new_gs = GimbalState.REWIND
                        idle_dur = 0.0
                        target_id = None
                        miss_count = 0

        elif old_gs == GimbalState.REWIND:
            if has_blobs:
                new_gs = (
                    GimbalState.TRACKING if _any_acquired(blobs, cfg) else GimbalState.ACQUIRING
                )
                idle_dur = 0.0
            elif at_limb:
                new_gs = GimbalState.TRACKING
                idle_dur = 0.0
                target_id = None

        if new_gs != old_gs:
            events.append(self._transition_event(old_gs, new_gs))

        request: GimbalRequest | None = None

        if new_gs == GimbalState.TRACKING and has_blobs and error_deg is not None:
            best = _select_best_target(blobs)
            target_id = best.blob_id
            if _rate_ok(last_cmd_time, now, cfg.retarget_rate_limit_hz):
                limit = gimbal.max_hw_slew_rate_deg_per_s
                az_rate = min(max(error_deg[0] * 1.0, -limit), limit)
                el_rate = min(max(error_deg[1] * 1.0, -limit), limit)
                request = GimbalRequest(
                    mode=GimbalCommandMode.RATE,
                    az_deg=az_rate,
                    el_deg=el_rate,
                    reason="tracking_target",
                )
                last_cmd_time = now

        elif new_gs == GimbalState.REWIND:
            if _rate_ok(last_cmd_time, now, cfg.retarget_rate_limit_hz):
                request = GimbalRequest(
                    mode=GimbalCommandMode.ABSOLUTE,
                    az_deg=0.0,
                    el_deg=gimbal.el_science_max_deg,
                    reason="rewind_to_limb",
                )
                last_cmd_time = now

        new_state = ArbiterState(
            gimbal_state=new_gs,
            tracked_blobs=blobs,
            idle_duration_s=idle_dur,
            last_command_time=last_cmd_time,
            current_target_id=target_id,
            miss_count=miss_count,
        )
        return new_state, request, events

    @staticmethod
    def _transition_event(from_state: GimbalState, to_state: GimbalState) -> TelemetryEventMsg:
        """Build the state_transition telemetry event for one arbiter transition.

        Args:
            from_state: The GimbalState before the transition.
            to_state: The GimbalState after the transition.

        Returns:
            A TelemetryEventMsg recording the from/to states for the controller subsystem.
        """
        return TelemetryEventMsg(
            msg_type=MessageType.TELEMETRY_EVENT,
            timestamp_utc=utc_now_iso(),
            subsystem="controller",
            event_name="state_transition",
            payload={"from": from_state.value, "to": to_state.value},
        )


def _any_acquired(
    blobs: tuple[BlobMeta, ...],
    cfg: ControllerConfig,
) -> bool:
    """Return True if any blob has persistence >= acquire threshold."""
    return any(b.persistence_count >= cfg.acquire_persistence_frames for b in blobs)


def _select_best_target(blobs: tuple[BlobMeta, ...]) -> BlobMeta:
    """Select best target: highest persistence, then confidence."""
    return max(
        blobs,
        key=lambda b: (b.persistence_count, b.mean_confidence),
    )


def _rate_ok(
    last_cmd_time: float,
    now: float,
    rate_hz: float,
) -> bool:
    """Check if enough time elapsed for a new command."""
    if rate_hz <= 0.0:
        return False
    return (now - last_cmd_time) >= (1.0 / rate_hz)
