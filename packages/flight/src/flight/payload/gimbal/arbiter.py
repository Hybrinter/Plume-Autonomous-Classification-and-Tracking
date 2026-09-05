"""Gimbal arbiter: TRACKING / REWIND / SAFE mode selection (pure).

The arbiter selects the rate-reference policy. It does not emit axis rates or
torque. SAFE latches until ground clears it. A plume moves the machine to
TRACKING immediately. Loss below the science limb after release persistence
enters REWIND. Arrival at the limb with no plume is TRACKING with r = 0.

Satisfies: REQ-AIML-GIMB-001 through 008, REQ-GIMB-HIGH-001 through 004
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from flight.libs.config import ControllerConfig, GimbalConfig
from flight.libs.messages import BlobMeta, TelemetryEventMsg, utc_now_iso
from flight.libs.types import GimbalCommandMode, GimbalState, MessageType
from flight.payload.gimbal.request import GimbalRequest


@dataclass(frozen=True)
class ArbiterState:
    """Immutable arbiter snapshot.

    Fields
    ------
    gimbal_state:
        TRACKING, REWIND, or SAFE.
    tracked_blobs:
        Blobs that survived safety gates on the last vision sample.
    current_target_id:
        blob_id of the tracked target, or None.
    miss_count:
        Consecutive vision samples with no blob while in TRACKING.
    """

    gimbal_state: GimbalState
    tracked_blobs: tuple[BlobMeta, ...]
    current_target_id: int | None
    miss_count: int = 0


class GimbalArbiter:
    """TRACKING / REWIND / SAFE gimbal arbiter. REQ-AIML-GIMB-008.

    step() is a pure function aside from timestamp strings on returned telemetry.
    GimbalArbiter holds no mutable instance state; ArbiterState threads externally.
    """

    def __init__(self, cfg: ControllerConfig, gimbal: GimbalConfig) -> None:
        """Hold controller thresholds and the science-limb elevation.

        Args:
            cfg: ControllerConfig supplying release persistence and limb arrival.
            gimbal: GimbalConfig supplying science-limb elevation.
        """
        self._cfg = cfg
        self._gimbal = gimbal

    def step(
        self,
        state: ArbiterState,
        blobs: tuple[BlobMeta, ...],
        now: float,
        safe_commanded: bool,
        safe_cleared: bool,
        el_deg: float | None,
        mode_flags: int = 0,
        vision_updated: bool = True,
    ) -> tuple[ArbiterState, GimbalRequest | None, list[TelemetryEventMsg]]:
        """Advance the mode machine by one outer tick.

        Parameters
        ----------
        state:
            Current immutable arbiter state.
        blobs:
            Gated, IoU-matched blobs from the latest vision sample (empty on coast).
        now:
            Monotonic seconds (unused for rates; kept for the pure-core signature).
        safe_commanded:
            True if a SAFE mode change was drained: latch SAFE and stow.
        safe_cleared:
            True if a non-SAFE mode change was drained: exit SAFE to TRACKING.
        el_deg:
            Current elevation in signed off-nadir degrees, or None.
        mode_flags:
            Inference mode_flags; any nonzero value latches SAFE.
        vision_updated:
            True when this tick consumed a vision sample. False on outer coast:
            miss_count is unchanged.

        Returns
        -------
        (new_state, request, telemetry_events)
            request is STOW on SAFE entry, otherwise None. The outer law owns r.
        """
        del now  # mode machine does not use time deltas
        cfg = self._cfg
        gimbal = self._gimbal
        old_gs = state.gimbal_state
        blobs_now = blobs if vision_updated else state.tracked_blobs
        has_plume = len(blobs_now) > 0
        events: list[TelemetryEventMsg] = []
        at_limb = el_deg is not None and el_deg >= gimbal.el_science_max_deg - cfg.limb_arrival_deg

        if (safe_commanded or mode_flags != 0) and old_gs != GimbalState.SAFE:
            new_state = replace(
                state,
                gimbal_state=GimbalState.SAFE,
                tracked_blobs=blobs_now,
                miss_count=0,
                current_target_id=None,
            )
            events.append(self._transition_event(old_gs, GimbalState.SAFE))
            stow_request = GimbalRequest(
                mode=GimbalCommandMode.STOW,
                el_deg=gimbal.stow_el_deg,
                reason="safe_entry_stow",
            )
            return new_state, stow_request, events

        if old_gs == GimbalState.SAFE:
            if safe_cleared:
                new_state = ArbiterState(
                    gimbal_state=GimbalState.TRACKING,
                    tracked_blobs=(),
                    current_target_id=None,
                    miss_count=0,
                )
                events.append(self._transition_event(GimbalState.SAFE, GimbalState.TRACKING))
                return new_state, None, events
            return replace(state, tracked_blobs=blobs_now), None, events

        new_gs = old_gs
        target_id = state.current_target_id
        miss_count = state.miss_count

        if old_gs == GimbalState.TRACKING:
            if not vision_updated:
                pass
            elif has_plume:
                miss_count = 0
                target_id = blobs_now[0].blob_id
            else:
                miss_count = state.miss_count + 1
                if miss_count >= cfg.release_persistence_frames:
                    if at_limb:
                        miss_count = 0
                        target_id = None
                    else:
                        new_gs = GimbalState.REWIND
                        target_id = None
                        miss_count = 0

        elif old_gs == GimbalState.REWIND:
            if has_plume:
                new_gs = GimbalState.TRACKING
                miss_count = 0
                target_id = blobs_now[0].blob_id
            elif at_limb:
                new_gs = GimbalState.TRACKING
                miss_count = 0
                target_id = None

        if new_gs != old_gs:
            events.append(self._transition_event(old_gs, new_gs))

        new_state = ArbiterState(
            gimbal_state=new_gs,
            tracked_blobs=blobs_now,
            current_target_id=target_id,
            miss_count=miss_count,
        )
        return new_state, None, events

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
