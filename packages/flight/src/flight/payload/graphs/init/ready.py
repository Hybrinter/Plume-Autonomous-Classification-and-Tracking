"""INIT READY node: request explicit verification, then IDLE once (pure).

READY emits one VERIFY_INIT intent scoped to the activation, then waits for an
InitVerificationResult on the current key. A VERIFIED result with nonempty
evidence emits exactly one INIT_COMPLETE request while the graph stays READY
until external graph selection. A denied or dropped request is not sent again.
A FAILED verification latches failed state,
raises GIMBAL_FAULT, and requests SAFE once.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

from dataclasses import replace

from flight.libs.types import FaultCode
from flight.payload.gimbal.request import InhibitReference
from flight.payload.graphs.base import (
    EffectIntent,
    EffectKind,
    InitVerificationStatus,
    NodeOutcome,
    SystemRequestIntent,
    TickInputs,
)
from flight.payload.graphs.init.state import InitNode, State
from flight.payload.graphs.parameters import GraphParameters


def step(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, NodeOutcome[InitNode]]:
    """Emit VERIFY_INIT once; on VERIFIED emit one INIT_COMPLETE request.

    Inputs:
        state: Current INIT state.
        inputs: Tick observations including an optional verification result.
        params: Graph parameters.

    Outputs:
        tuple[State, NodeOutcome[InitNode]]: New state and the inhibit
            reference with the off policy, intents, and at most one
            INIT_COMPLETE or SAFE request.
    """
    policy = params.default_policy(enabled=False)
    verify_issued = EffectKind.VERIFY_INIT in state.issued
    effects: tuple[EffectIntent, ...] = ()
    if not verify_issued:
        state = replace(state, issued=state.issued | {EffectKind.VERIFY_INIT})
        effects = (
            EffectIntent(
                activation_key=state.activation_key,
                effect_id=EffectKind.VERIFY_INIT.value,
                kind=EffectKind.VERIFY_INIT,
            ),
        )
    if state.failed:
        return state, NodeOutcome(
            reference=InhibitReference(reason="init_failed"),
            policy=policy,
            effects=effects,
        )
    verification = inputs.verification
    if (
        not verify_issued
        or verification is None
        or verification.activation_key != state.activation_key
        or verification.status is InitVerificationStatus.PENDING
    ):
        return state, NodeOutcome(
            reference=InhibitReference(reason="init_ready"),
            policy=policy,
            effects=effects,
        )
    if verification.status is InitVerificationStatus.FAILED:
        state = replace(state, failed=True)
        return state, NodeOutcome(
            reference=InhibitReference(reason="init_verification_failed"),
            policy=policy,
            effects=effects,
            system_request=SystemRequestIntent.SAFE,
            faults=(FaultCode.GIMBAL_FAULT,),
        )
    if not verification.evidence_id:
        return state, NodeOutcome(
            reference=InhibitReference(reason="init_ready"),
            policy=policy,
            effects=effects,
        )
    system_request = None
    if not state.requested_idle:
        state = replace(state, requested_idle=True)
        system_request = SystemRequestIntent.INIT_COMPLETE
    return state, NodeOutcome(
        reference=InhibitReference(reason="init_ready"),
        policy=policy,
        effects=effects,
        system_request=system_request,
    )
