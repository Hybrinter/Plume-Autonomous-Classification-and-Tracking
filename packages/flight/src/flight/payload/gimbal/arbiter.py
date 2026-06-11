"""
Gimbal arbiter state machine for PACT.

Implements the four-state + SAFE arbiter that governs all gimbal commands.
The arbiter is a pure function: GimbalArbiter.step() has no side effects and holds
no references to queues, hardware, or I/O. All state transitions are returned as
TelemetryEventMsg values to be dispatched by the caller (the payload app shell).

The arbiter emits a typed GimbalRequest (not a bus message): RATE during TRACKING,
ABSOLUTE for the SCAN raster, STOW on SAFE entry. SAFE is latched in the arbiter and
commanded/cleared by ModeChangeMsg flags (safe_commanded/safe_cleared) drained by the
shell; pointing error is supplied as boresight-relative degrees by the caller, killing
the old absolute-centroid PIXEL_TO_DEG conversion.

Satisfies: REQ-AIML-GIMB-001 through 008, REQ-GIMB-HIGH-001 through 004
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from flight.libs.config import ControllerConfig
from flight.libs.messages import (
    BlobMeta,
    InferenceResultMsg,
    TelemetryEventMsg,
    utc_now_iso,
)
from flight.libs.types import GimbalCommandMode, GimbalState, MessageType
from flight.payload.gimbal.request import GimbalRequest

_SCAN_LIMIT_DEG: float = 30.0  # azimuth raster half-span before the scan reverses direction


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
        Unix timestamp of the most recent command issued. Used by the rate limiter.
    current_target_id:
        blob_id of the blob currently being tracked, or None if not in TRACKING.
    scan_pan_deg:
        Current azimuth pan position during SCAN mode (absolute degrees).
    scan_direction:
        Sign (+1.0 / -1.0) of the SCAN raster sweep; flips at the +-_SCAN_LIMIT_DEG edge.
    miss_count:
        TRACKING release-hysteresis counter: consecutive frames with no blob while in
        TRACKING. TRACKING releases to IDLE only once miss_count reaches
        release_persistence_frames; any blob resets it to 0.
    """

    gimbal_state: GimbalState
    tracked_blobs: tuple[BlobMeta, ...]
    idle_duration_s: float
    last_command_time: float  # Unix timestamp; 0.0 if no command has been issued yet
    current_target_id: int | None
    scan_pan_deg: float = 0.0  # current pan position during SCAN mode
    scan_direction: float = 1.0  # SCAN raster sweep sign; flips at the travel edges
    miss_count: int = 0  # TRACKING release-hysteresis counter


class GimbalArbiter:
    """Four-state + SAFE gimbal arbiter. REQ-AIML-GIMB-008.

    State Machine
    -------------
    States
        IDLE        No blob in view, or confidence gate has not been met.
        ACQUIRING   Blob detected and above threshold, but persistence < acquire frames.
        TRACKING    Blob held for >= acquire_persistence_frames consecutive frames.
        SCAN        IDLE for > scan_entry_idle_seconds; execute an absolute raster slew.
        SAFE        Fault/ground-induced; latched. Only exit is a cleared-mode signal.

    Transitions (all logged as TelemetryEventMsg)
        IDLE       -> ACQUIRING  : blob detected, persistence < acquire threshold
        ACQUIRING  -> TRACKING   : persistence >= acquire_persistence_frames
        TRACKING   -> IDLE       : no blobs for release_persistence_frames frames
        IDLE       -> SCAN       : idle_duration_s > scan_entry_idle_seconds
        SCAN       -> ACQUIRING  : blob detected above threshold
        any        -> SAFE       : safe_commanded or InferenceResultMsg.mode_flags != 0
        SAFE       -> IDLE       : safe_cleared (ground recovery)

    Design Notes
    ------------
    - step() is a **pure function**: no I/O, no queue access, no random, no time.time().
      The caller supplies `now` (monotonic seconds) and the boresight-relative error so
      the function is fully deterministic and trivially testable without mocking.
    - GimbalArbiter itself is stateless -- it stores no mutable instance state. ArbiterState
      is threaded externally through the app loop.
    - SAFE latches: while in SAFE, no further requests are produced and blobs are ignored
      until safe_cleared returns the machine to IDLE.
    - The TRACKING command is a proportional fallback (gain 1.0 / s) on the boresight error;
      control.py refines it via the LQR once the estimator is initialized. The SCAN raster
      is an ABSOLUTE pan that reverses at +-_SCAN_LIMIT_DEG (the old delta scan never
      reversed).
    """

    def __init__(self, cfg: ControllerConfig) -> None:
        """Hold the immutable controller config for thresholds, rates, and limits.

        Args:
            cfg: The ControllerConfig supplying gates, persistence, and slew limits.
        """
        self._cfg = cfg

    def step(
        self,
        state: ArbiterState,
        result: InferenceResultMsg,
        error_deg: tuple[float, float] | None,
        now: float,
        safe_commanded: bool,
        safe_cleared: bool,
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

        Returns
        -------
        (new_state, request, telemetry_events)
            new_state        : Updated ArbiterState after this step.
            request          : GimbalRequest to issue, or None.
            telemetry_events : Zero or more TelemetryEventMsg for state transitions.
        """
        cfg = self._cfg
        old_gs = state.gimbal_state
        blobs = result.blobs
        has_blobs = len(blobs) > 0
        events: list[TelemetryEventMsg] = []

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
                    scan_pan_deg=state.scan_pan_deg,
                    scan_direction=state.scan_direction,
                    miss_count=0,
                )
                events.append(self._transition_event(GimbalState.SAFE, GimbalState.IDLE))
                return new_state, None, events
            return replace(state, tracked_blobs=blobs), None, events

        new_gs = old_gs
        idle_dur = state.idle_duration_s
        target_id = state.current_target_id
        scan_pan = state.scan_pan_deg
        scan_direction = state.scan_direction
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
            if has_blobs:
                miss_count = 0
            else:
                miss_count = state.miss_count + 1
                if miss_count >= cfg.release_persistence_frames:
                    new_gs = GimbalState.IDLE
                    idle_dur = 0.0
                    target_id = None
                    miss_count = 0

        elif old_gs == GimbalState.SCAN:
            if has_blobs:
                new_gs = (
                    GimbalState.TRACKING if _any_acquired(blobs, cfg) else GimbalState.ACQUIRING
                )
                idle_dur = 0.0

        if new_gs != old_gs:
            events.append(self._transition_event(old_gs, new_gs))

        request: GimbalRequest | None = None

        if new_gs == GimbalState.TRACKING and has_blobs and error_deg is not None:
            best = _select_best_target(blobs)
            target_id = best.blob_id
            if _rate_ok(last_cmd_time, now, cfg.retarget_rate_limit_hz):
                limit = cfg.max_slew_rate_deg_per_s
                az_rate = min(max(error_deg[0] * 1.0, -limit), limit)
                el_rate = min(max(error_deg[1] * 1.0, -limit), limit)
                request = GimbalRequest(
                    mode=GimbalCommandMode.RATE,
                    az_deg=az_rate,
                    el_deg=el_rate,
                    reason="tracking_target",
                )
                last_cmd_time = now

        elif new_gs == GimbalState.SCAN:
            if _rate_ok(last_cmd_time, now, cfg.retarget_rate_limit_hz):
                scan_pan = scan_pan + scan_direction * cfg.scan_slew_rate_deg_per_s * (
                    1.0 / cfg.retarget_rate_limit_hz
                )
                if scan_pan > _SCAN_LIMIT_DEG:
                    scan_pan = _SCAN_LIMIT_DEG
                    scan_direction = -1.0
                elif scan_pan < -_SCAN_LIMIT_DEG:
                    scan_pan = -_SCAN_LIMIT_DEG
                    scan_direction = 1.0
                request = GimbalRequest(
                    mode=GimbalCommandMode.ABSOLUTE,
                    az_deg=scan_pan,
                    el_deg=0.0,
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
            scan_direction=scan_direction,
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
