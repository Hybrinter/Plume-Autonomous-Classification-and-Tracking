"""Gimbal arbiter: TRACKING / REWIND / SAFE mode selection (pure).

The arbiter selects the rate-reference policy. It does not emit axis rates or
torque. SAFE latches until ground clears it. A plume moves the machine to
TRACKING immediately. A loss below the science limb enters REWIND after the
first of a bounded empty-result release or observation-age timeout. Arrival at
the limb with no plume is TRACKING with r = 0.

Satisfies: REQ-AIML-GIMB-001 through 008, REQ-GIMB-HIGH-001 through 004
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from flight.libs.config import ArbiterConfig, GimbalConfig
from flight.libs.messages import BlobMeta, TelemetryEventMsg
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
    aggregate_live:
        Whether an accepted aggregate is present or is still within its
        bounded prediction-only coast. This is deliberately independent of
        connected-component IDs.
    last_observation_s:
        Shutter time of the most recent accepted aggregate, or None before one
        has been seen. It is retained after loss for age telemetry.
    loss_handled:
        True after the aggregate has exhausted its coast. It prevents a
        limbed REWIND from immediately re-entering REWIND every outer tick.
    miss_count:
        Consecutive vision samples with no blob while in TRACKING.
    rewind_entered_s:
        Monotonic seconds when REWIND was entered, or None when not in REWIND.
    """

    gimbal_state: GimbalState
    tracked_blobs: tuple[BlobMeta, ...]
    current_target_id: int | None
    miss_count: int = 0
    aggregate_live: bool = False
    last_observation_s: float | None = None
    loss_handled: bool = False
    rewind_entered_s: float | None = None


class GimbalArbiter:
    """TRACKING / REWIND / SAFE gimbal arbiter. REQ-AIML-GIMB-008.

    step() is a pure function aside from timestamp strings on returned telemetry.
    GimbalArbiter holds no mutable instance state; ArbiterState threads externally.
    """

    def __init__(self, cfg: ArbiterConfig, gimbal: GimbalConfig) -> None:
        """Hold arbiter thresholds and the science-limb elevation.

        Args:
            cfg: ArbiterConfig supplying release persistence and limb arrival.
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
        observation_t_s: float | None = None,
        coast_permitted: bool = True,
        timestamp_utc: str = "",
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
            miss_count is unchanged, but observation age still advances.
        observation_t_s:
            Shutter time of the consumed vision sample. Defaults to ``now`` for
            a direct caller that has no separate shutter timestamp.
        coast_permitted:
            Estimator-uncertainty seam. A later estimator selection can revoke
            prediction-only coasting without changing the mode machine.
        timestamp_utc:
            Injected ISO stamp for transition telemetry. Empty uses a blank stamp.

        Returns
        -------
        (new_state, request, telemetry_events)
            request is STOW on SAFE entry, otherwise None. The outer law owns r.
        """
        cfg = self._cfg
        gimbal = self._gimbal
        old_gs = state.gimbal_state
        blobs_now = blobs if vision_updated else state.tracked_blobs
        # Old association records survive an outer coast so that the next vision
        # sample can match them. They are not a new observation and must never
        # refresh aggregate liveness or observation age.
        has_plume = vision_updated and len(blobs) > 0
        events: list[TelemetryEventMsg] = []
        at_limb = el_deg is not None and el_deg >= gimbal.el_science_max_deg - cfg.limb_arrival_deg

        if (safe_commanded or mode_flags != 0) and old_gs != GimbalState.SAFE:
            new_state = replace(
                state,
                gimbal_state=GimbalState.SAFE,
                tracked_blobs=blobs_now,
                miss_count=0,
                current_target_id=None,
                aggregate_live=False,
                last_observation_s=None,
                loss_handled=False,
                rewind_entered_s=None,
            )
            events.append(self._transition_event(old_gs, GimbalState.SAFE, timestamp_utc))
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
                    aggregate_live=False,
                    last_observation_s=None,
                    loss_handled=False,
                    rewind_entered_s=None,
                )
                events.append(
                    self._transition_event(GimbalState.SAFE, GimbalState.TRACKING, timestamp_utc)
                )
                return new_state, None, events
            return replace(state, tracked_blobs=blobs_now, aggregate_live=False), None, events

        new_gs = old_gs
        # Kept as a compatibility-only telemetry field. Aggregate validity and
        # control do not select or depend on an individual component ID.
        target_id: int | None = None
        miss_count = state.miss_count
        last_observation_s = state.last_observation_s
        loss_handled = state.loss_handled

        if has_plume:
            # An accepted aggregate enters TRACKING immediately. Component IDs
            # remain association metadata only; the aggregate owns liveness.
            last_observation_s = now if observation_t_s is None else observation_t_s
            miss_count = 0
            loss_handled = False
            if old_gs is GimbalState.REWIND:
                new_gs = GimbalState.TRACKING
        elif vision_updated:
            miss_count = state.miss_count + 1

        age_s = None if last_observation_s is None else max(0.0, now - last_observation_s)
        empty_release = vision_updated and miss_count >= cfg.release_persistence_frames
        age_release = age_s is not None and age_s >= cfg.max_observation_age_s
        coast_exhausted = (empty_release or age_release or not coast_permitted) and not has_plume

        if old_gs is GimbalState.TRACKING and coast_exhausted and not loss_handled:
            loss_handled = True
            target_id = None
            if at_limb:
                miss_count = 0
            else:
                new_gs = GimbalState.REWIND
                miss_count = 0
        elif old_gs is GimbalState.REWIND and not has_plume and at_limb:
            new_gs = GimbalState.TRACKING
            miss_count = 0

        aggregate_live = has_plume or (
            last_observation_s is not None
            and not loss_handled
            and coast_permitted
            and not empty_release
            and not age_release
        )

        if new_gs != old_gs:
            events.append(self._transition_event(old_gs, new_gs, timestamp_utc))

        if new_gs is GimbalState.REWIND:
            rewind_entered_s = now if old_gs is not GimbalState.REWIND else state.rewind_entered_s
            if rewind_entered_s is None:
                rewind_entered_s = now
        else:
            rewind_entered_s = None

        new_state = ArbiterState(
            gimbal_state=new_gs,
            tracked_blobs=blobs_now,
            current_target_id=target_id,
            miss_count=miss_count,
            aggregate_live=aggregate_live,
            last_observation_s=last_observation_s,
            loss_handled=loss_handled,
            rewind_entered_s=rewind_entered_s,
        )
        return new_state, None, events

    @staticmethod
    def _transition_event(
        from_state: GimbalState, to_state: GimbalState, timestamp_utc: str
    ) -> TelemetryEventMsg:
        """Build the state_transition telemetry event for one arbiter transition.

        Args:
            from_state: The GimbalState before the transition.
            to_state: The GimbalState after the transition.
            timestamp_utc: Injected ISO timestamp (not wall clock).

        Returns:
            A TelemetryEventMsg recording the from/to states for the controller subsystem.
        """
        return TelemetryEventMsg(
            msg_type=MessageType.TELEMETRY_EVENT,
            timestamp_utc=timestamp_utc,
            subsystem="controller",
            event_name="state_transition",
            payload={"from": from_state.value, "to": to_state.value},
        )
