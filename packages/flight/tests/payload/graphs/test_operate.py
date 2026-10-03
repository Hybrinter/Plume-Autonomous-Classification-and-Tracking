"""Tests for the OPERATE graph: tracking, hunts, hold, and ingest rules."""

import math

import numpy as np
from flight.libs.config import EphemerisConfig
from flight.libs.types import (
    ActivationKey,
    FaultCode,
)
from flight.payload.gimbal.request import InhibitReference, PoseReference, RateReference
from flight.payload.graphs import operate
from flight.payload.graphs.base import SystemRequestIntent
from flight.payload.graphs.operate.state import HoldState, State, TargetState
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.records import (
    HealthSample,
    IssSample,
)

from .support import BlobBuilder, TickBuilder, VisionBuilder


def _state(
    key: ActivationKey,
    params: GraphParameters,
    node: operate.OperateNode,
    *,
    hold_reason: operate.HoldReason = operate.HoldReason.LIMB_WAIT,
    hold_target_rad: float | None = None,
    loss_handled: bool = False,
    rewind_entered_s: float | None = None,
) -> State:
    """Build a seeded OPERATE state at one node with cold tracking memory."""
    return State(
        activation_key=key,
        node=node,
        tracked_blobs=(),
        aggregate_live=False,
        last_observation_s=None,
        miss_count=0,
        loss_handled=loss_handled,
        rewind_entered_s=rewind_entered_s,
        residual=params.residual_filter.initial_state(),
        residual_history=params.residual_filter.initial_history(),
        target=TargetState(
            r_cog_ecef_m=None,
            last_exposure_us=1000.0,
            last_theta_los=0.0,
            last_omega_t_nom=0.0,
            last_omega_az_nom=0.0,
            last_omega_scene_el=0.0,
        ),
        hold=HoldState(reason=hold_reason, target_rad=hold_target_rad),
        seen_vision=(),
    )


_STALE = HealthSample(feedback_valid=False, inhibit_confirmed=False, contained=False)
_CONTAINED = HealthSample(feedback_valid=True, inhibit_confirmed=False, contained=True)


def _iss(dt_s: float = 0.0) -> IssSample:
    """Circular-LEO IssSample `dt_s` after the ephemeris epoch."""
    eph = EphemerisConfig()
    radius = 6_378_137.0 + 400_000.0
    speed = math.sqrt(eph.mu_m3_s2 / radius)
    theta = (speed / radius) * dt_s
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return IssSample(
        r_m=(radius * cos_t, radius * sin_t, 0.0),
        v_m_s=(-speed * sin_t, speed * cos_t, 0.0),
        utc_s=eph.epoch_utc_s + dt_s,
    )


def test_initial_state_is_tracking_cold(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Cold OPERATE starts TRACKING with no ancestry and no residual."""
    state = operate.initial_state(tick(0.0, key), params)
    assert state.node is operate.OperateNode.TRACKING
    assert state.tracked_blobs == ()
    assert state.aggregate_live is False
    assert state.residual.has_measurement is False
    assert state.hold.target_rad is None


def test_accepted_blob_tracks_with_rate_reference(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """An above-boresight blob yields TRACKING and a positive rate."""
    state = operate.initial_state(tick(0.0, key), params)
    sample = vision(
        0.0,
        key,
        blobs=(blob(1, (612.0, 512.0 - 70.0)),),
        theta_g_rad=math.radians(20.0),
        iss=_iss(),
    )
    new_state, outcome = operate.step(
        state, tick(0.02, key, encoder_angle_rad=math.radians(20.0), vision=sample), params
    )
    assert new_state.node is operate.OperateNode.TRACKING
    assert new_state.aggregate_live is True
    assert new_state.residual.has_measurement is True
    assert isinstance(outcome.outcome.reference, RateReference)
    assert outcome.outcome.reference.rate_rad_s > 0.0


def test_tracking_does_not_require_navigation(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """A valid visual aggregate commands tracking with unknown navigation."""
    state = operate.initial_state(tick(0.0, key), params)
    sample = vision(0.0, key, blobs=(blob(),), theta_g_rad=math.radians(20.0), iss=None)
    new_state, outcome = operate.step(
        state, tick(0.02, key, encoder_angle_rad=math.radians(20.0), vision=sample), params
    )
    assert new_state.aggregate_live is True
    assert isinstance(outcome.outcome.reference, RateReference)
    assert outcome.outcome.reference.rate_rad_s > 0.0


def test_coast_exhaustion_away_from_limb_rewinds(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """A stalled pipeline in TRACKING commits COAST_EXHAUSTED to REWIND."""
    state = operate.initial_state(tick(0.0, key), params)
    sample = vision(0.0, key, blobs=(blob(),), theta_g_rad=math.radians(20.0))
    state, _ = operate.step(
        state, tick(0.02, key, encoder_angle_rad=math.radians(20.0), vision=sample), params
    )
    now = params.config.controller.operate.max_observation_age_s + 0.04
    new_state, outcome = operate.step(
        state, tick(now, key, encoder_angle_rad=math.radians(20.0)), params
    )
    assert new_state.node is operate.OperateNode.REWIND
    assert outcome.transition is not None
    assert outcome.transition.trigger.value == "coast_exhausted"
    assert new_state.loss_handled is True
    assert new_state.miss_count == 0
    assert isinstance(outcome.outcome.reference, RateReference)
    assert outcome.outcome.reference.rate_rad_s > 0.0


def test_delayed_plume_keeps_newest_observation_time(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """A late plume inside the replay window does not age the live aggregate."""
    from flight.payload.tracking.residual import ObservationDisposition

    theta = math.radians(20.0)
    plume = blob(1, (612.0, 442.0))
    state = operate.initial_state(tick(0.0, key, encoder_angle_rad=theta), params)
    first = vision(0.0, key, blobs=(plume,), theta_g_rad=theta)
    state, _ = operate.step(state, tick(0.0, key, encoder_angle_rad=theta, vision=first), params)
    for t_s in (0.02, 0.04):
        state, _ = operate.step(state, tick(t_s, key, encoder_angle_rad=theta), params)
    newer_t = 0.06
    newer = vision(newer_t, key, blobs=(plume,), theta_g_rad=theta)
    state, _ = operate.step(
        state, tick(newer_t, key, encoder_angle_rad=theta, vision=newer), params
    )
    assert state.last_observation_s == newer_t
    delayed_t = 0.04
    delayed = vision(delayed_t, key, blobs=(plume,), theta_g_rad=theta)
    arrival = 0.08
    age_limit = params.config.controller.operate.max_observation_age_s
    assert arrival - delayed_t <= params.residual_filter.rewind_horizon_s
    assert arrival - delayed_t <= age_limit
    state, _ = operate.step(
        state, tick(arrival, key, encoder_angle_rad=theta, vision=delayed), params
    )
    assert state.last_observation_s == newer_t
    assert state.vision_disposition is ObservationDisposition.ACCEPTED
    assert any(
        obs.frame_id == delayed.sample.frame_id
        for obs in state.residual_history.vision_observations
    )
    now = delayed_t + age_limit
    assert now - newer_t < age_limit
    state, outcome = operate.step(state, tick(now, key, encoder_angle_rad=theta), params)
    assert state.node is operate.OperateNode.TRACKING
    assert outcome.transition is None
    assert state.aggregate_live is True
    assert state.last_observation_s == newer_t


def test_coast_exhaustion_at_limb_holds(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """Coast exhaustion at the science limb holds as LIMB_WAIT."""
    angle = math.radians(params.config.gimbal.el_science_max_deg)
    state = operate.initial_state(tick(0.0, key, encoder_angle_rad=angle), params)
    sample = vision(0.0, key, blobs=(blob(),), theta_g_rad=angle)
    state, _ = operate.step(state, tick(0.02, key, encoder_angle_rad=angle, vision=sample), params)
    now = params.config.controller.operate.max_observation_age_s + 0.04
    new_state, outcome = operate.step(state, tick(now, key, encoder_angle_rad=angle), params)
    assert new_state.node is operate.OperateNode.HOLD
    assert new_state.hold.reason is operate.HoldReason.LIMB_WAIT
    assert new_state.hold.target_rad == angle
    assert new_state.miss_count == 0
    assert new_state.aggregate_live is False
    assert new_state.last_rate_decision is None
    assert isinstance(outcome.outcome.reference, PoseReference)


def test_empty_frames_release_by_persistence(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """Enough consecutive empty accepted frames exhaust the coast."""
    state = operate.initial_state(tick(0.0, key), params)
    sample = vision(0.0, key, blobs=(blob(),), theta_g_rad=math.radians(20.0))
    state, _ = operate.step(
        state, tick(0.02, key, encoder_angle_rad=math.radians(20.0), vision=sample), params
    )
    persist = params.config.controller.operate.release_persistence_frames
    for index in range(persist):
        t_s = 0.04 + index * 0.02
        empty = vision(t_s, key, blobs=(), theta_g_rad=math.radians(20.0))
        state, outcome = operate.step(
            state,
            tick(t_s, key, encoder_angle_rad=math.radians(20.0), vision=empty),
            params,
        )
    assert state.node is operate.OperateNode.REWIND
    assert outcome.transition is not None
    assert state.miss_count == 0


def test_release_frame_id_seen_once(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """The releasing empty frame is recorded once in the seen window."""
    state = operate.initial_state(tick(0.0, key), params)
    sample = vision(0.0, key, blobs=(blob(),), theta_g_rad=math.radians(20.0))
    state, _ = operate.step(
        state, tick(0.02, key, encoder_angle_rad=math.radians(20.0), vision=sample), params
    )
    persist = params.config.controller.operate.release_persistence_frames
    last_empty = None
    for index in range(persist):
        t_s = 0.04 + index * 0.02
        last_empty = vision(t_s, key, blobs=(), theta_g_rad=math.radians(20.0))
        state, _ = operate.step(
            state,
            tick(t_s, key, encoder_angle_rad=math.radians(20.0), vision=last_empty),
            params,
        )
    assert last_empty is not None
    ids = [frame_id for frame_id, _ in state.seen_vision]
    assert ids.count(last_empty.sample.frame_id) == 1


def test_manual_hold_never_claims_live_aggregate(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """Accepted vision inside a MANUAL hold stays metadata, not a live target."""
    state = _state(
        key,
        params,
        operate.OperateNode.HOLD,
        hold_reason=operate.HoldReason.MANUAL,
        hold_target_rad=math.radians(20.0),
    )
    sample = vision(0.0, key, blobs=(blob(),), theta_g_rad=math.radians(20.0))
    new_state, _ = operate.step(
        state, tick(0.02, key, encoder_angle_rad=math.radians(20.0), vision=sample), params
    )
    assert new_state.node is operate.OperateNode.HOLD
    assert new_state.aggregate_live is False
    assert len(new_state.tracked_blobs) == len(sample.sample.blobs) > 0
    assert new_state.last_rate_decision is None


def test_tracking_records_rate_decision_terms(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """TRACKING stores the exact RateDecision with its flags and terms."""
    state = operate.initial_state(tick(0.0, key), params)
    sample = vision(0.0, key, blobs=(blob(1, (612.0, 442.0)),), theta_g_rad=math.radians(20.0))
    new_state, _ = operate.step(
        state, tick(0.02, key, encoder_angle_rad=math.radians(20.0), vision=sample), params
    )
    decision = new_state.last_rate_decision
    assert decision is not None
    assert (
        decision.requested_relative_rate_rad_s
        == decision.commanded_rate_rad_s - decision.scene_rate_rad_s
    )
    assert decision.smear_limit_rad_s > 0.0
    assert decision.hardware_limited is False
    assert decision.science_limited is False


def test_rewind_reacquires_to_tracking(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """A plume mid-REWIND commits VISION_ACQUIRED and cold-starts residual."""
    state = operate.initial_state(tick(0.0, key), params)
    sample = vision(0.0, key, blobs=(blob(1),), theta_g_rad=math.radians(20.0))
    state, _ = operate.step(
        state, tick(0.02, key, encoder_angle_rad=math.radians(20.0), vision=sample), params
    )
    now = params.config.controller.operate.max_observation_age_s + 0.04
    state, _ = operate.step(state, tick(now, key, encoder_angle_rad=math.radians(20.0)), params)
    assert state.node is operate.OperateNode.REWIND
    planted = np.array([0.15, 0.04], dtype=np.float64)
    from dataclasses import replace

    state = replace(state, residual=replace(state.residual, x=planted, has_measurement=True))
    acquire = vision(now + 0.02, key, blobs=(blob(2),), theta_g_rad=math.radians(20.0))
    new_state, outcome = operate.step(
        state,
        tick(now + 0.04, key, encoder_angle_rad=math.radians(20.0), vision=acquire),
        params,
    )
    assert new_state.node is operate.OperateNode.TRACKING
    assert outcome.transition is not None
    assert outcome.transition.trigger.value == "vision_acquired"
    assert not np.allclose(new_state.residual.x, planted)


def test_vision_acquires_at_limb_hunt_over_limb_edge(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """Accepted vision outranks the limb edge while hunting."""
    angle = math.radians(params.config.gimbal.el_science_max_deg)
    state = _state(
        key,
        params,
        operate.OperateNode.REWIND,
        loss_handled=True,
        rewind_entered_s=0.0,
    )
    sample = vision(0.02, key, blobs=(blob(),), theta_g_rad=angle)
    new_state, outcome = operate.step(
        state, tick(0.04, key, encoder_angle_rad=angle, vision=sample), params
    )
    assert new_state.node is operate.OperateNode.TRACKING
    assert outcome.transition is not None
    assert outcome.transition.trigger.value == "vision_acquired"


def test_hunt_at_limb_arrives_at_hold(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A hunt reaching the science limb commits LIMB_ARRIVAL to HOLD."""
    angle = math.radians(params.config.gimbal.el_science_max_deg)
    state = _state(
        key,
        params,
        operate.OperateNode.FAST_REWIND,
        loss_handled=True,
        rewind_entered_s=0.0,
    )
    new_state, outcome = operate.step(state, tick(0.04, key, encoder_angle_rad=angle), params)
    assert new_state.node is operate.OperateNode.HOLD
    assert outcome.transition is not None
    assert outcome.transition.trigger.value == "limb_arrival"
    assert new_state.hold.target_rad == angle


def test_rewind_timer_promotes_to_fast(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """REWIND past the sharp window commits TIMER_EXPIRED to FAST_REWIND."""
    state = _state(
        key,
        params,
        operate.OperateNode.REWIND,
        loss_handled=True,
        rewind_entered_s=0.0,
    )
    horizon = params.config.controller.outer.rewind_sharp_max_s
    new_state, outcome = operate.step(
        state, tick(horizon + 0.01, key, encoder_angle_rad=math.radians(20.0)), params
    )
    assert new_state.node is operate.OperateNode.FAST_REWIND
    assert outcome.transition is not None
    assert outcome.transition.trigger.value == "timer_expired"


def test_manual_hold_ignores_vision(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """A MANUAL hold never auto-exits on vision."""
    state = _state(
        key,
        params,
        operate.OperateNode.HOLD,
        hold_reason=operate.HoldReason.MANUAL,
        hold_target_rad=math.radians(20.0),
    )
    sample = vision(0.0, key, blobs=(blob(),), theta_g_rad=math.radians(20.0))
    new_state, outcome = operate.step(
        state, tick(0.02, key, encoder_angle_rad=math.radians(20.0), vision=sample), params
    )
    assert new_state.node is operate.OperateNode.HOLD
    assert outcome.transition is None


def test_limb_hold_reacquires_to_tracking(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """A LIMB_WAIT hold commits VISION_ACQUIRED back to TRACKING."""
    state = _state(
        key,
        params,
        operate.OperateNode.HOLD,
        hold_reason=operate.HoldReason.LIMB_WAIT,
        hold_target_rad=math.radians(45.0),
        loss_handled=True,
    )
    sample = vision(0.0, key, blobs=(blob(),), theta_g_rad=math.radians(45.0))
    new_state, outcome = operate.step(
        state, tick(0.02, key, encoder_angle_rad=math.radians(45.0), vision=sample), params
    )
    assert new_state.node is operate.OperateNode.TRACKING
    assert outcome.transition is not None


def test_mode_flags_request_safe_without_moving(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """A flagged vision sample faults INFERENCE_NAN and requests SAFE."""
    state = operate.initial_state(tick(0.0, key), params)
    sample = vision(0.0, key, blobs=(blob(),), mode_flags=1)
    new_state, outcome = operate.step(
        state, tick(0.02, key, encoder_angle_rad=0.0, vision=sample), params
    )
    assert new_state.node is operate.OperateNode.TRACKING
    assert outcome.outcome.faults == (FaultCode.INFERENCE_NAN,)
    assert outcome.outcome.system_request is SystemRequestIntent.SAFE
    assert isinstance(outcome.outcome.reference, InhibitReference)
    assert outcome.outcome.policy.imaging.acquisition_enabled is False
    assert outcome.outcome.policy.inference.enabled is False


def test_stale_context_and_duplicate_frames_do_not_refresh(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """Wrong-key, stale, and duplicate frames produce no liveness refresh."""
    state = operate.initial_state(tick(0.0, key), params)
    good = vision(0.0, key, blobs=(blob(),), theta_g_rad=math.radians(20.0))
    state, _ = operate.step(
        state, tick(0.02, key, encoder_angle_rad=math.radians(20.0), vision=good), params
    )
    last_obs = state.last_observation_s
    wrong_key = vision(0.02, ActivationKey(epoch="test", sequence=99), blobs=(blob(),))
    state2, _ = operate.step(
        state, tick(0.04, key, encoder_angle_rad=math.radians(20.0), vision=wrong_key), params
    )
    assert state2.last_observation_s == last_obs
    stale_rev = vision(0.02, key, blobs=(blob(),), policy_revision=7)
    state3, _ = operate.step(
        state, tick(0.04, key, encoder_angle_rad=math.radians(20.0), vision=stale_rev), params
    )
    assert state3.last_observation_s == last_obs
    duplicate = vision(0.0, key, frame_id=good.sample.frame_id, blobs=(blob(),))
    state4, _ = operate.step(
        state, tick(0.04, key, encoder_angle_rad=math.radians(20.0), vision=duplicate), params
    )
    assert state4.last_observation_s == last_obs
    future = vision(10.0, key, blobs=(blob(),))
    state5, _ = operate.step(
        state, tick(0.04, key, encoder_angle_rad=math.radians(20.0), vision=future), params
    )
    assert state5.last_observation_s == last_obs


def test_stale_feedback_inhibits(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Stale or missing feedback inhibits motion and keeps imaging enabled."""
    state = operate.initial_state(tick(0.0, key), params)
    new_state, outcome = operate.step(state, tick(1.0, key, encoder_t_s=0.0, health=_STALE), params)
    assert new_state == state
    assert isinstance(outcome.outcome.reference, InhibitReference)
    assert outcome.outcome.reference.reason == "stale_feedback"
    assert outcome.transition is None
    assert outcome.outcome.policy.imaging.acquisition_enabled is True
    assert outcome.outcome.policy.inference.enabled is True
    missing_state, missing = operate.step(state, tick(0.02, key, encoder_angle_rad=None), params)
    assert missing_state == state
    assert isinstance(missing.outcome.reference, InhibitReference)
    assert missing.outcome.reference.reason == "stale_feedback"
    assert missing.outcome.policy.imaging.acquisition_enabled is True
    assert missing.outcome.policy.inference.enabled is True


def test_contained_requests_safe(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Contained health inhibits and emits one SAFE intent."""
    state = operate.initial_state(tick(0.0, key), params)
    new_state, outcome = operate.step(state, tick(0.02, key, health=_CONTAINED), params)
    assert new_state == state
    assert outcome.outcome.system_request is SystemRequestIntent.SAFE
    assert outcome.outcome.policy.imaging.acquisition_enabled is False
    assert outcome.outcome.policy.inference.enabled is False


def test_same_inputs_same_outputs(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """Stepping identical state and inputs twice is deterministic."""
    state = operate.initial_state(tick(0.0, key), params)
    sample = vision(0.0, key, blobs=(blob(),), theta_g_rad=math.radians(20.0))
    inputs = tick(0.02, key, encoder_angle_rad=math.radians(20.0), vision=sample)
    first = operate.step(state, inputs, params)
    second = operate.step(state, inputs, params)
    assert np.array_equal(first[0].residual.x, second[0].residual.x)
    assert np.array_equal(first[0].residual.P, second[0].residual.P)
    assert first[0].node is second[0].node
    assert first[0].tracked_blobs == second[0].tracked_blobs
    assert first[0].aggregate_live is second[0].aggregate_live
    assert first[1].outcome.reference == second[1].outcome.reference


def test_rewind_uses_boresight_not_stored_cog(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """REWIND predicts from boresight and clears the stored plume CoG."""
    state = operate.initial_state(tick(0.0, key), params)
    iss = _iss()
    sample = vision(0.0, key, blobs=(blob(),), theta_g_rad=math.radians(20.0), iss=iss)
    state, _ = operate.step(
        state,
        tick(0.02, key, encoder_angle_rad=math.radians(20.0), vision=sample, navigation=iss),
        params,
    )
    assert state.target.r_cog_ecef_m is not None
    now = params.config.controller.operate.max_observation_age_s + 0.04
    state, outcome = operate.step(
        state,
        tick(now, key, encoder_angle_rad=math.radians(20.0), navigation=iss),
        params,
    )
    assert state.node is operate.OperateNode.REWIND
    assert state.target.r_cog_ecef_m is None
