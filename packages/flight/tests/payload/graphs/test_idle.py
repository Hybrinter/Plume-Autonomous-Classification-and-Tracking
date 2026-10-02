"""Tests for the IDLE graph: pose capture on fresh feedback only."""

from flight.payload.gimbal.request import InhibitReference, PoseReference
from flight.payload.graphs import idle
from flight.payload.graphs.base import SystemRequestIntent
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.records import ActivationKey, HealthSample

from .support import TickBuilder

_STALE = HealthSample(feedback_valid=False, inhibit_confirmed=False, contained=False)
_CONTAINED = HealthSample(feedback_valid=True, inhibit_confirmed=False, contained=True)


def test_initial_state_captures_pose_when_fresh(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A fresh encoder at entry becomes the held pose target."""
    state = idle.initial_state(tick(0.0, key, encoder_angle_rad=0.5), params)
    assert state.node is idle.IdleNode.HOLD
    assert state.target_rad == 0.5


def test_initial_state_without_fresh_feedback(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """No encoder at entry leaves no target."""
    state = idle.initial_state(tick(0.0, key, encoder_angle_rad=None), params)
    assert state.target_rad is None


def test_step_holds_captured_pose(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """HOLD emits a PoseReference to the captured angle under the pose envelope."""
    state = idle.initial_state(tick(0.0, key, encoder_angle_rad=0.5), params)
    new_state, outcome = idle.step(state, tick(0.02, key, encoder_angle_rad=0.5), params)
    assert new_state.target_rad == 0.5
    assert isinstance(outcome.outcome.reference, PoseReference)
    assert outcome.outcome.reference.target_rad == 0.5
    assert outcome.outcome.policy.imaging.acquisition_enabled is False


def test_first_fresh_sample_sets_target(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A later first fresh encoder sample sets the held target."""
    state = idle.initial_state(tick(0.0, key, encoder_angle_rad=None), params)
    new_state, outcome = idle.step(state, tick(0.02, key, encoder_angle_rad=0.25), params)
    assert new_state.target_rad == 0.25
    assert isinstance(outcome.outcome.reference, PoseReference)


def test_stale_feedback_inhibits(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Stale feedback inhibits and keeps the state unchanged."""
    state = idle.initial_state(tick(0.0, key, encoder_angle_rad=0.5), params)
    inputs = tick(1.0, key, encoder_angle_rad=0.5, encoder_t_s=0.0, health=_STALE)
    new_state, outcome = idle.step(state, inputs, params)
    assert new_state == state
    assert isinstance(outcome.outcome.reference, InhibitReference)


def test_contained_requests_safe(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Contained health inhibits and emits one SAFE intent."""
    state = idle.initial_state(tick(0.0, key, encoder_angle_rad=0.5), params)
    inputs = tick(0.02, key, encoder_angle_rad=0.5, health=_CONTAINED)
    _, outcome = idle.step(state, inputs, params)
    assert isinstance(outcome.outcome.reference, InhibitReference)
    assert outcome.outcome.system_request is SystemRequestIntent.SAFE
    assert outcome.transition is None
