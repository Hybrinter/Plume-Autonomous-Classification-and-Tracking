"""Tests for the INIT graph: ordered effects, verification, request once."""

from flight.libs.types import FaultCode
from flight.payload.gimbal.request import InhibitReference, PoseReference
from flight.payload.graphs import init
from flight.payload.graphs.base import (
    EffectKind,
    EffectResult,
    EffectStatus,
    InitVerificationResult,
    InitVerificationStatus,
    SystemRequestIntent,
)
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.records import ActivationKey, HealthSample

from .support import TickBuilder

_STALE = HealthSample(feedback_valid=False, inhibit_confirmed=False, contained=False)


def _result(
    key: ActivationKey,
    kind: EffectKind,
    status: EffectStatus,
    *,
    effect_id: str | None = None,
    fault: FaultCode = FaultCode.NONE,
    evidence_id: str = "",
) -> EffectResult:
    """Build one typed effect result for the current node kind."""
    return EffectResult(
        activation_key=key,
        effect_id=kind.value if effect_id is None else effect_id,
        kind=kind,
        status=status,
        fault=fault,
        evidence_id=evidence_id,
    )


def test_initial_state_is_selftest(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Cold INIT starts at SELFTEST with nothing issued or requested."""
    state = init.initial_state(tick(0.0, key), params)
    assert state.node is init.InitNode.SELFTEST
    assert state.issued == frozenset()
    assert state.requested_idle is False


def test_each_effect_intent_emitted_once(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """The node emits its effect intent exactly once."""
    state = init.initial_state(tick(0.0, key), params)
    state, first = init.step(state, tick(0.0, key), params)
    kinds = [effect.kind for effect in first.outcome.effects]
    assert kinds == [EffectKind.SELFTEST]
    state, second = init.step(state, tick(0.02, key), params)
    assert second.outcome.effects == ()


def test_unsolicited_result_does_not_advance(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A SUCCEEDED result never issued or for the wrong key is ignored."""
    state = init.initial_state(tick(0.0, key), params)
    unsolicited = _result(key, EffectKind.HOME, EffectStatus.SUCCEEDED)
    new_state, outcome = init.step(state, tick(0.0, key, effect_results=(unsolicited,)), params)
    assert new_state.node is init.InitNode.SELFTEST
    assert outcome.transition is None


def test_pending_result_does_not_advance(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A PENDING result for the issued effect stays at SELFTEST."""
    state = init.initial_state(tick(0.0, key), params)
    state, _ = init.step(state, tick(0.0, key), params)
    pending = _result(key, EffectKind.SELFTEST, EffectStatus.PENDING)
    new_state, _ = init.step(state, tick(0.02, key, effect_results=(pending,)), params)
    assert new_state.node is init.InitNode.SELFTEST


def test_success_advances_one_edge(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A SUCCEEDED result for the issued effect commits one edge."""
    state = init.initial_state(tick(0.0, key), params)
    state, _ = init.step(state, tick(0.0, key), params)
    done = _result(key, EffectKind.SELFTEST, EffectStatus.SUCCEEDED)
    new_state, outcome = init.step(state, tick(0.02, key, effect_results=(done,)), params)
    assert new_state.node is init.InitNode.MODEL_LOAD
    assert outcome.transition is not None
    assert outcome.transition.trigger.value == "effect_completed"
    assert outcome.outcome.events


def test_failed_latches_fault_and_safe_once(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A FAILED effect result latches failed, one fault, one SAFE request."""
    state = init.initial_state(tick(0.0, key), params)
    state, _ = init.step(state, tick(0.0, key), params)
    failed = _result(key, EffectKind.SELFTEST, EffectStatus.FAILED)
    new_state, outcome = init.step(state, tick(0.02, key, effect_results=(failed,)), params)
    assert new_state.failed is True
    assert outcome.outcome.system_request is SystemRequestIntent.SAFE
    assert outcome.outcome.faults == (FaultCode.GIMBAL_FAULT,)
    _, second = init.step(new_state, tick(0.04, key), params)
    assert second.outcome.system_request is None
    assert second.outcome.faults == ()


def test_home_requires_nonempty_evidence(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A SUCCEEDED HOME result without evidence_id does not advance."""
    state = init.initial_state(tick(0.0, key), params)
    state, _ = init.step(state, tick(0.0, key), params)
    state, _ = init.step(
        state,
        tick(
            0.02, key, effect_results=(_result(key, EffectKind.SELFTEST, EffectStatus.SUCCEEDED),)
        ),
        params,
    )
    state, _ = init.step(state, tick(0.04, key), params)
    state, _ = init.step(
        state,
        tick(
            0.06, key, effect_results=(_result(key, EffectKind.MODEL_LOAD, EffectStatus.SUCCEEDED),)
        ),
        params,
    )
    assert state.node is init.InitNode.HOME
    bare = _result(key, EffectKind.HOME, EffectStatus.SUCCEEDED, evidence_id="")
    new_state, outcome = init.step(state, tick(0.08, key, effect_results=(bare,)), params)
    assert new_state.node is init.InitNode.HOME
    assert outcome.transition is None


def test_ready_verified_emits_init_complete_once(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """READY requests INIT_COMPLETE once on VERIFIED evidence; stays READY."""
    state = init.initial_state(tick(0.0, key), params)
    state, _ = init.step(state, tick(0.0, key), params)
    for t_s, kind in (
        (0.02, EffectKind.SELFTEST),
        (0.04, EffectKind.MODEL_LOAD),
        (0.06, EffectKind.HOME),
    ):
        state, _ = init.step(state, tick(t_s, key), params)
        result = _result(
            key, kind, EffectStatus.SUCCEEDED, evidence_id="ev" if kind is EffectKind.HOME else ""
        )
        state, _ = init.step(state, tick(t_s + 0.01, key, effect_results=(result,)), params)
    assert state.node is init.InitNode.READY
    assert EffectKind.VERIFY_INIT in state.issued
    state, ready_out = init.step(state, tick(0.10, key), params)
    assert ready_out.outcome.effects == ()
    verified = InitVerificationResult(
        activation_key=key,
        status=InitVerificationStatus.VERIFIED,
        evidence_id="verifier:ok",
    )
    new_state, outcome = init.step(state, tick(0.12, key, verification=verified), params)
    assert outcome.outcome.system_request is SystemRequestIntent.INIT_COMPLETE
    assert new_state.node is init.InitNode.READY
    assert new_state.requested_idle is True
    _, second = init.step(new_state, tick(0.14, key, verification=verified), params)
    assert second.outcome.system_request is None


def test_ready_pending_waits(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """READY with no verification result stays pending and inhibited."""
    state = init.State(
        activation_key=key,
        node=init.InitNode.READY,
        issued=frozenset({EffectKind.VERIFY_INIT}),
        failed=False,
        requested_idle=False,
        home_target_rad=0.0,
    )
    new_state, outcome = init.step(state, tick(0.0, key), params)
    assert new_state.node is init.InitNode.READY
    assert outcome.outcome.system_request is None
    assert isinstance(outcome.outcome.reference, InhibitReference)


def test_ready_failed_verification_latches(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A FAILED verification raises GIMBAL_FAULT and requests SAFE once."""
    state = init.State(
        activation_key=key,
        node=init.InitNode.READY,
        issued=frozenset({EffectKind.VERIFY_INIT}),
        failed=False,
        requested_idle=False,
        home_target_rad=0.0,
    )
    failed = InitVerificationResult(
        activation_key=key, status=InitVerificationStatus.FAILED, evidence_id=""
    )
    new_state, outcome = init.step(state, tick(0.0, key, verification=failed), params)
    assert new_state.failed is True
    assert outcome.outcome.faults == (FaultCode.GIMBAL_FAULT,)
    assert outcome.outcome.system_request is SystemRequestIntent.SAFE


def _ready_state(key: ActivationKey, params: GraphParameters) -> init.State:
    """A READY state with VERIFY_INIT not yet issued."""
    import math

    return init.State(
        activation_key=key,
        node=init.InitNode.READY,
        issued=frozenset(),
        failed=False,
        requested_idle=False,
        home_target_rad=math.radians(params.config.gimbal.home_el_deg),
    )


def test_ready_ignores_same_tick_verification_before_intent(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A VERIFIED result on the intent-issuing tick cannot request IDLE."""
    state = _ready_state(key, params)
    verified = InitVerificationResult(
        activation_key=key,
        status=InitVerificationStatus.VERIFIED,
        evidence_id="verifier:ok",
    )
    new_state, outcome = init.step(state, tick(0.0, key, verification=verified), params)
    assert EffectKind.VERIFY_INIT in new_state.issued
    assert outcome.outcome.system_request is None
    assert new_state.requested_idle is False
    newer, second = init.step(new_state, tick(0.02, key, verification=verified), params)
    assert second.outcome.system_request is SystemRequestIntent.INIT_COMPLETE
    assert newer.node is init.InitNode.READY


def test_ready_ignores_unsolicited_verification_key(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A verification result under another activation key is ignored."""
    state = _ready_state(key, params)
    state, _ = init.step(state, tick(0.0, key), params)
    stale = InitVerificationResult(
        activation_key=ActivationKey(epoch="test", sequence=99),
        status=InitVerificationStatus.VERIFIED,
        evidence_id="verifier:ok",
    )
    new_state, outcome = init.step(state, tick(0.02, key, verification=stale), params)
    assert new_state.requested_idle is False
    assert outcome.outcome.system_request is None


def test_home_stale_feedback_defers_intent_until_fresh(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """HOME does not issue its intent while encoder feedback is stale."""
    import math

    state = init.State(
        activation_key=key,
        node=init.InitNode.HOME,
        issued=frozenset(),
        failed=False,
        requested_idle=False,
        home_target_rad=math.radians(params.config.gimbal.home_el_deg),
    )
    new_state, outcome = init.step(state, tick(1.0, key, encoder_t_s=0.0, health=_STALE), params)
    assert new_state.issued == frozenset()
    assert outcome.outcome.effects == ()
    newer, later = init.step(new_state, tick(1.02, key), params)
    assert EffectKind.HOME in newer.issued
    assert any(effect.kind is EffectKind.HOME for effect in later.outcome.effects)


def test_stale_key_result_does_not_advance(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A SUCCEEDED result under an older activation key is ignored."""
    state = init.initial_state(tick(0.0, key), params)
    state, _ = init.step(state, tick(0.0, key), params)
    stale = _result(
        ActivationKey(epoch="test", sequence=0), EffectKind.SELFTEST, EffectStatus.SUCCEEDED
    )
    new_state, _ = init.step(state, tick(0.02, key, effect_results=(stale,)), params)
    assert new_state.node is init.InitNode.SELFTEST


def test_duplicate_result_commits_at_most_one_edge(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Duplicate SUCCEEDED results in one batch commit only one edge."""
    state = init.initial_state(tick(0.0, key), params)
    state, _ = init.step(state, tick(0.0, key), params)
    result = _result(key, EffectKind.SELFTEST, EffectStatus.SUCCEEDED)
    new_state, outcome = init.step(state, tick(0.02, key, effect_results=(result, result)), params)
    assert new_state.node is init.InitNode.MODEL_LOAD
    assert outcome.transition is not None
    assert sum(1 for event in outcome.outcome.events if event.event_name == "node_transition") == 1


def test_failed_result_beats_succeeded_in_same_batch(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A FAILED receipt wins over a same-batch SUCCEEDED for the issued effect."""
    state = init.initial_state(tick(0.0, key), params)
    state, _ = init.step(state, tick(0.0, key), params)
    failed = _result(
        key, EffectKind.SELFTEST, EffectStatus.FAILED, fault=FaultCode.GIMBAL_ENCODER_INVALID
    )
    succeeded = _result(key, EffectKind.SELFTEST, EffectStatus.SUCCEEDED)
    for batch in ((succeeded, failed), (failed, succeeded)):
        new_state, outcome = init.step(state, tick(0.02, key, effect_results=batch), params)
        assert new_state.failed is True
        assert new_state.node is init.InitNode.SELFTEST
        assert outcome.outcome.faults == (FaultCode.GIMBAL_ENCODER_INVALID,)
        assert outcome.outcome.system_request is SystemRequestIntent.SAFE


def test_home_poses_to_configured_target(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """HOME emits a PoseReference to the configured home elevation."""
    import math

    state = init.State(
        activation_key=key,
        node=init.InitNode.HOME,
        issued=frozenset({EffectKind.HOME}),
        failed=False,
        requested_idle=False,
        home_target_rad=math.radians(params.config.gimbal.home_el_deg),
    )
    _, outcome = init.step(state, tick(0.0, key), params)
    assert isinstance(outcome.outcome.reference, PoseReference)
    assert outcome.outcome.reference.target_rad == math.radians(params.config.gimbal.home_el_deg)
