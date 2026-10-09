"""OPERATE graph state, node, and hold vocabulary (pure).

TRACKING keeps the residual estimate and scene terms; the hunts hold no target
CoG; HOLD is limb-wait or manual. Shared vision bookkeeping lives here once
rather than per node.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001, REQ-AIML-GIMB-008.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

from flight.libs.messages import BlobMeta
from flight.libs.types import ActivationKey
from flight.payload.gimbal.outer import RateDecision
from flight.payload.gimbal.pointing import pinhole_error_rad
from flight.payload.gimbal.safety import apply_confidence_gate, apply_min_area_gate
from flight.payload.graphs.base import TickInputs
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.records import CapturedVision
from flight.payload.tracking import match_blobs
from flight.payload.tracking.residual import (
    ObservationDisposition,
    ResidualHistory,
    ResidualState,
)


class OperateNode(Enum):
    """OPERATE graph node vocabulary."""

    TRACKING = "tracking"
    REWIND = "rewind"
    FAST_REWIND = "fast_rewind"
    HOLD = "hold"


class HoldReason(Enum):
    """Why HOLD was entered: limb wait (auto-exits) or manual (never)."""

    LIMB_WAIT = "limb_wait"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class HoldState:
    """Hold occupancy: reason and the captured or commanded pose.

    Attributes:
        reason: LIMB_WAIT exits on accepted vision; MANUAL never auto-exits.
        target_rad: Captured or commanded hold elevation, rad, or None until
            fresh feedback establishes one.
    """

    reason: HoldReason
    target_rad: float | None


@dataclass(frozen=True, slots=True)
class TargetState:
    """Stored CoG and last scene-rate terms owned by the TRACKING node.

    Attributes:
        r_cog_ecef_m: Last good CoG Earth point, ECEF meters. None in hunts.
        last_exposure_us: Last live exposure (rewind smear cap).
        last_theta_los: Last predictor elevation, rad.
        last_omega_t_nom: Last co-rotating elevation rate, rad/s.
        last_omega_az_nom: Last unactuated optical-azimuth rate, rad/s.
        last_omega_scene_el: Last elevation scene rate used for smear, rad/s.
    """

    r_cog_ecef_m: tuple[float, float, float] | None
    last_exposure_us: float
    last_theta_los: float
    last_omega_t_nom: float
    last_omega_az_nom: float
    last_omega_scene_el: float


@dataclass(frozen=True, slots=True)
class State:
    """OPERATE graph state threaded across ticks.

    Attributes:
        activation_key: Activation this state belongs to.
        node: Current node.
        tracked_blobs: Gated, IoU-matched blobs from the last accepted sample.
        aggregate_live: Accepted aggregate present or inside bounded coast.
        last_observation_s: Newest shutter time among accepted aggregates.
        miss_count: Consecutive accepted samples with no blob while tracking.
        loss_handled: True after coast exhaustion was committed once.
        rewind_entered_s: Monotonic time the hunt began, kept through
            FAST_REWIND, or None outside hunts.
        hunt_timeout_latched: True after a hunt that never reached the limb
            requested SAFE once.
        residual: Latest two-state residual Kalman estimate.
        residual_history: Timestamped residual events and posterior anchor.
        target: Stored CoG and scene-rate terms.
        hold: Hold occupancy while in HOLD.
        seen_vision: (frame_id, shutter) pairs inside the acceptance window.
        last_rate_decision: Most recent computed `RateDecision`, or None on
            the pose path.
        last_e_az: Last unactuated optical-azimuth error, rad; retained when
            no accepted sample carries a centroid.
        vision_disposition: Residual-history disposition of the current
            accepted vision sample, or None.
    """

    activation_key: ActivationKey
    node: OperateNode
    tracked_blobs: tuple[BlobMeta, ...]
    aggregate_live: bool
    last_observation_s: float | None
    miss_count: int
    loss_handled: bool
    rewind_entered_s: float | None
    residual: ResidualState
    residual_history: ResidualHistory
    target: TargetState
    hold: HoldState
    seen_vision: tuple[tuple[str, float], ...]
    last_rate_decision: RateDecision | None = None
    last_e_az: float = 0.0
    vision_disposition: ObservationDisposition | None = None
    hunt_timeout_latched: bool = False


def vision_window_s(params: GraphParameters) -> float:
    """Acceptance window: max(observation age ceiling, residual history horizon)."""
    controller = params.config.controller
    return max(controller.operate.max_observation_age_s, controller.residual.rewind_horizon_s)


def accept_vision(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> CapturedVision | None:
    """Resolve the raw captured sample into the effective vision sample.

    Rejects a sample whose context key or policy revision does not match, whose
    shutter is future, expired, or nonfinite, or whose frame id was already
    seen. An accepted sample is returned with blobs gated and IoU-matched
    against the stored ancestry, the area-weighted union centroid, and the
    pinhole boresight error filled in.

    Inputs:
        state: Current OPERATE state supplying ancestry and the seen window.
        inputs: TickInputs carrying the raw CapturedVision.
        params: GraphParameters supplying gates, revision, and geometry.

    Outputs:
        CapturedVision | None: The effective sample, or None when rejected or
            absent. Rejected samples produce no liveness or miss refresh.
    """
    vision = inputs.vision
    if vision is None:
        return None
    if vision.context.activation_key != state.activation_key:
        return None
    if vision.context.policy_revision != params.policy_revision:
        return None
    sample = vision.sample
    now_s = inputs.now_s
    window = vision_window_s(params)
    if not math.isfinite(sample.t_s):
        return None
    if sample.t_s > now_s + 1.0e-12 or sample.t_s < now_s - window - 1.0e-12:
        return None
    if any(frame_id == sample.frame_id for frame_id, _ in state.seen_vision):
        return None
    controller = params.config.controller
    gated = apply_confidence_gate(sample.blobs, controller.vision.confidence_gate)
    gated = apply_min_area_gate(gated, controller.vision.min_blob_area_px)
    matched = match_blobs(state.tracked_blobs, gated, controller.vision.blob_iou_match_threshold)
    z_v: float | None = None
    p_cog: tuple[float, float] | None = None
    if matched:
        # All accepted components form one visible-plume aggregate.  Do not
        # select a component by ID or order: each component centroid is
        # weighted by its accepted pixel area, which is equivalent to the
        # union-pixel centroid for disjoint connected components.
        total_area = sum(blob.pixel_area for blob in matched)
        p_cog = (
            sum(blob.pixel_area * blob.centroid_raw[0] for blob in matched) / total_area,
            sum(blob.pixel_area * blob.centroid_raw[1] for blob in matched) / total_area,
        )
        camera = params.camera
        _e_az, z_v = pinhole_error_rad(
            p_cog,
            camera.width_px,
            camera.height_px,
            camera.pixel_pitch_m,
            camera.focal_length_m,
        )
    return replace(vision, sample=replace(sample, blobs=matched, z_v=z_v, p_cog=p_cog))


def bookkeep_vision(
    state: State,
    vision: CapturedVision | None,
    inputs: TickInputs,
    params: GraphParameters,
) -> State:
    """Apply one accepted (or absent) vision sample to shared bookkeeping.

    An accepted sample with blobs clears the miss counter, keeps the newest
    shutter time, and stores the matched ancestry. An accepted empty sample
    advances the miss count without refreshing the observation time.
    A rejected or absent sample changes neither. The seen-window prunes by the
    larger of the observation-age ceiling and the residual history horizon.

    Inputs:
        state: Current OPERATE state (post-commit loss/rewind fields applied).
        vision: Effective sample from accept_vision, or None.
        inputs: Tick observations carrying the monotonic time.
        params: GraphParameters supplying release persistence and age limits.

    Outputs:
        State: Bookkeeping fields updated; aggregate_live recomputed.
    """
    controller = params.config.controller
    now_s = inputs.now_s
    miss_count = state.miss_count
    last_observation_s = state.last_observation_s
    tracked = state.tracked_blobs
    loss_handled = state.loss_handled
    if vision is not None:
        sample = vision.sample
        if sample.blobs:
            miss_count = 0
            last_observation_s = (
                sample.t_s if last_observation_s is None else max(last_observation_s, sample.t_s)
            )
            loss_handled = False
            tracked = sample.blobs
        else:
            miss_count = state.miss_count + 1
            tracked = ()
    age_s = None if last_observation_s is None else max(0.0, now_s - last_observation_s)
    empty_release = (
        vision is not None and miss_count >= controller.operate.release_persistence_frames
    )
    age_release = age_s is not None and age_s >= controller.operate.max_observation_age_s
    has_plume = vision is not None and len(vision.sample.blobs) > 0
    aggregate_live = has_plume or (
        last_observation_s is not None
        and not loss_handled
        and not empty_release
        and not age_release
    )
    window = vision_window_s(params)
    seen = tuple(pair for pair in state.seen_vision if pair[1] >= now_s - window - 1.0e-12)
    if vision is not None:
        seen = seen + ((vision.sample.frame_id, vision.sample.t_s),)
    return replace(
        state,
        tracked_blobs=tracked,
        aggregate_live=aggregate_live,
        last_observation_s=last_observation_s,
        miss_count=miss_count,
        loss_handled=loss_handled,
        seen_vision=seen,
    )


def acquire_resets_residual(
    previous_node: OperateNode,
    previous_aggregate_live: bool,
    previous_blob_ids: frozenset[int],
    new_blob_ids: frozenset[int],
) -> bool:
    """True when entering TRACKING with an acquire that needs a cold residual.

    A reset is an acquire, not a loss: first blob from cold, a blob entering
    TRACKING from a hunt, or a tracked set with no overlapping blob_id versus
    the previous tracked set. An empty new set never resets.
    """
    if not new_blob_ids:
        return False
    if previous_node in (OperateNode.REWIND, OperateNode.FAST_REWIND):
        return True
    if not previous_aggregate_live:
        return True
    return bool(previous_blob_ids) and previous_blob_ids.isdisjoint(new_blob_ids)
