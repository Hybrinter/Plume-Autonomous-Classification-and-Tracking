"""Tests for the SAFE graph: inhibition only, never a pose."""

from flight.libs.types import ActivationKey
from flight.payload.gimbal.request import InhibitReference
from flight.payload.graphs import safe
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.records import HealthSample

from .support import TickBuilder

_CONTAINED = HealthSample(feedback_valid=True, inhibit_confirmed=True, contained=True)


def test_initial_state_is_inhibited(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Cold SAFE state is the single INHIBITED node."""
    state = safe.initial_state(tick(0.0, key), params)
    assert state.node is safe.SafeNode.INHIBITED


def test_step_always_inhibits_even_with_fresh_feedback(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """SAFE emits inhibit regardless of feedback; never a pose reference."""
    state = safe.initial_state(tick(0.0, key), params)
    new_state, outcome = safe.step(state, tick(0.02, key), params)
    assert new_state == state
    assert isinstance(outcome.outcome.reference, InhibitReference)
    assert outcome.outcome.policy.imaging.acquisition_enabled is False


def test_step_keeps_key_on_mismatched_activation(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A foreign activation key does not retarget SAFE and still inhibits."""
    state = safe.initial_state(tick(0.0, key), params)
    other = ActivationKey(epoch="other", sequence=key.sequence + 1)
    new_state, outcome = safe.step(state, tick(0.02, other), params)
    assert new_state.activation_key == key
    assert isinstance(outcome.outcome.reference, InhibitReference)


def test_step_inhibits_when_contained(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Contained health still yields only the inhibit reference."""
    state = safe.initial_state(tick(0.0, key), params)
    _, outcome = safe.step(state, tick(0.02, key, health=_CONTAINED), params)
    assert isinstance(outcome.outcome.reference, InhibitReference)
