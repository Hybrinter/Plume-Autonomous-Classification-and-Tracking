"""Tests for the STOW graph: bounded move, verified hold, latched timeout."""

from flight.libs.types import FaultCode
from flight.payload.gimbal.request import InhibitReference, StowReference
from flight.payload.graphs import stow
from flight.payload.graphs.base import SystemRequestIntent
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.records import ActivationKey, HealthSample

from .support import TickBuilder

_CONFIRMED = HealthSample(feedback_valid=True, inhibit_confirmed=True, contained=False)
_NO_CONFIRM = HealthSample(feedback_valid=True, inhibit_confirmed=False, contained=False)
_STALE = HealthSample(feedback_valid=False, inhibit_confirmed=False, contained=False)


def test_initial_state_is_moving(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Cold STOW state enters MOVING at the first tick time."""
    state = stow.initial_state(tick(1.0, key), params)
    assert state.node is stow.StowNode.MOVING
    assert state.entered_s == 1.0


def test_moving_emits_bounded_stow_reference(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """MOVING emits the configured stow target under the stow envelope."""
    state = stow.initial_state(tick(0.0, key), params)
    _, outcome = stow.step(state, tick(0.02, key), params)
    ref = outcome.outcome.reference
    assert isinstance(ref, StowReference)
    assert ref.envelope.omega_max_rad_s == params.stow_envelope.omega_max_rad_s
    assert ref.timeout_s == params.config.gimbal.xeryon.stow_timeout_s


def test_completion_evidence_commits_to_held(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """stow_complete with fresh confirmed feedback commits VERIFIED_STABLE."""
    state = stow.initial_state(tick(0.0, key), params)
    inputs = tick(0.02, key, health=_CONFIRMED, stow_complete=True)
    new_state, outcome = stow.step(state, inputs, params)
    assert new_state.node is stow.StowNode.HELD
    assert outcome.transition is not None
    assert outcome.transition.trigger.value == "verified_stable"
    assert outcome.outcome.events
    assert isinstance(outcome.outcome.reference, InhibitReference)


def test_completion_without_inhibit_confirm_stays_moving(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """stow_complete without confirmed inhibit does not commit the edge."""
    state = stow.initial_state(tick(0.0, key), params)
    inputs = tick(0.02, key, health=_NO_CONFIRM, stow_complete=True)
    new_state, outcome = stow.step(state, inputs, params)
    assert new_state.node is stow.StowNode.MOVING
    assert outcome.transition is None


def test_held_stays_held(params: GraphParameters, tick: TickBuilder, key: ActivationKey) -> None:
    """HELD keeps inhibiting with no further transitions."""
    state = stow.initial_state(tick(0.0, key), params)
    held, _ = stow.step(state, tick(0.02, key, health=_CONFIRMED, stow_complete=True), params)
    newer, outcome = stow.step(held, tick(0.04, key, health=_NO_CONFIRM), params)
    assert newer.node is stow.StowNode.HELD
    assert outcome.transition is None
    assert isinstance(outcome.outcome.reference, InhibitReference)


def test_timeout_latches_once(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """The bounded window expiring emits one fault and one SAFE request."""
    timeout = params.config.gimbal.xeryon.stow_timeout_s
    state = stow.initial_state(tick(0.0, key), params)
    new_state, outcome = stow.step(state, tick(timeout + 0.01, key), params)
    assert new_state.timeout_latched is True
    assert FaultCode.GIMBAL_SAFETY_TIMEOUT in outcome.outcome.faults
    assert outcome.outcome.system_request is SystemRequestIntent.SAFE
    _, second = stow.step(new_state, tick(timeout + 0.04, key), params)
    assert second.outcome.faults == ()
    assert second.outcome.system_request is None


def test_stale_feedback_inhibits_moving(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """MOVING without fresh feedback inhibits."""
    state = stow.initial_state(tick(0.0, key), params)
    _, outcome = stow.step(state, tick(0.02, key, health=_STALE), params)
    assert isinstance(outcome.outcome.reference, InhibitReference)
