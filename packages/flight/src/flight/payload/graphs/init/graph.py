"""INIT graph: sequenced self-test, model load, home, and ready (pure).

Each node issues one activation-scoped effect intent; an EFFECT_COMPLETED edge
commits only on a SUCCEEDED result whose key, kind, and id match an intent
already issued from the current node (HOME additionally requires nonempty
arrival evidence). FAILED results latch the failed flag with one fault and one
SAFE request. READY waits on InitVerificationResult for the current activation
and emits one INIT_COMPLETE request; nothing here grants local IDLE.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

import math
from dataclasses import replace

from flight.libs.types import FaultCode
from flight.payload.gimbal.request import InhibitReference
from flight.payload.graphs.base import (
    Edge,
    EdgeTrigger,
    EffectKind,
    EffectStatus,
    GraphId,
    GraphOutcome,
    GraphSpec,
    NodeOutcome,
    SystemRequestIntent,
    TickInputs,
    transition_event,
)
from flight.payload.graphs.init import home, model_load, ready, selftest
from flight.payload.graphs.init.state import InitNode, State
from flight.payload.graphs.parameters import GraphParameters

EDGES: tuple[Edge[InitNode], ...] = (
    Edge(
        source=InitNode.SELFTEST,
        target=InitNode.MODEL_LOAD,
        trigger=EdgeTrigger.EFFECT_COMPLETED,
    ),
    Edge(
        source=InitNode.MODEL_LOAD,
        target=InitNode.HOME,
        trigger=EdgeTrigger.EFFECT_COMPLETED,
    ),
    Edge(
        source=InitNode.HOME,
        target=InitNode.READY,
        trigger=EdgeTrigger.EFFECT_COMPLETED,
    ),
)

_NODE_EFFECT: dict[InitNode, EffectKind] = {
    InitNode.SELFTEST: EffectKind.SELFTEST,
    InitNode.MODEL_LOAD: EffectKind.MODEL_LOAD,
    InitNode.HOME: EffectKind.HOME,
    InitNode.READY: EffectKind.VERIFY_INIT,
}


def spec(params: GraphParameters) -> GraphSpec[InitNode]:
    """Declare the INIT graph.

    Inputs:
        params: Graph parameters supplying the default policy.

    Outputs:
        GraphSpec[InitNode]: The four-node effect chain.
    """
    policy = params.default_policy(enabled=False)
    return GraphSpec(
        graph_id=GraphId.INIT,
        nodes=(InitNode.SELFTEST, InitNode.MODEL_LOAD, InitNode.HOME, InitNode.READY),
        initial=InitNode.SELFTEST,
        edges=EDGES,
        imaging=policy.imaging,
        inference=policy.inference,
    )


def initial_state(inputs: TickInputs, params: GraphParameters) -> State:
    """Cold INIT state at SELFTEST with the configured home target.

    Inputs:
        inputs: First tick observations under this activation.
        params: Graph parameters supplying the home pose.

    Outputs:
        State: SELFTEST with nothing issued or requested.
    """
    return State(
        activation_key=inputs.activation_key,
        node=InitNode.SELFTEST,
        issued=frozenset(),
        failed=False,
        requested_idle=False,
        home_target_rad=math.radians(params.config.gimbal.home_el_deg),
    )


def _inhibit(
    state: State,
    params: GraphParameters,
    reason: str,
    system_request: SystemRequestIntent | None = None,
    faults: tuple[FaultCode, ...] = (),
) -> tuple[State, GraphOutcome[InitNode]]:
    """Same-state inhibit outcome helper."""
    return state, GraphOutcome(
        node=state.node,
        outcome=NodeOutcome(
            reference=InhibitReference(reason=reason),
            policy=params.default_policy(enabled=False),
            system_request=system_request,
            faults=faults,
        ),
    )


def _node_step(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, NodeOutcome[InitNode]]:
    """Dispatch to the current node's step function."""
    match state.node:
        case InitNode.SELFTEST:
            return selftest.step(state, inputs, params)
        case InitNode.MODEL_LOAD:
            return model_load.step(state, inputs, params)
        case InitNode.HOME:
            return home.step(state, inputs, params)
        case InitNode.READY:
            return ready.step(state, inputs, params)


def step(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, GraphOutcome[InitNode]]:
    """Advance the INIT graph by one tick.

    Inputs:
        state: Current INIT state.
        inputs: Tick observations including effect and verification results.
        params: Graph parameters.

    Outputs:
        tuple[State, GraphOutcome[InitNode]]: New state and the committed
            outcome; at most one edge commits per tick.
    """
    if inputs.activation_key != state.activation_key:
        return _inhibit(state, params, "activation_mismatch")
    if inputs.health.contained:
        return _inhibit(state, params, "contained", system_request=SystemRequestIntent.SAFE)
    if state.failed:
        return _inhibit(state, params, "init_failed")

    expected_kind = _NODE_EFFECT[state.node]
    applicable = [
        result
        for result in inputs.effect_results
        if result.activation_key == state.activation_key
        and result.kind is expected_kind
        and result.kind in state.issued
        and result.effect_id == result.kind.value
        and result.status is not EffectStatus.PENDING
    ]
    failed = [result for result in applicable if result.status is EffectStatus.FAILED]
    if failed:
        result = failed[0]
        new_state = replace(state, failed=True)
        fault = result.fault
        if fault is FaultCode.NONE:
            fault = FaultCode.GIMBAL_FAULT
        return new_state, GraphOutcome(
            node=state.node,
            outcome=NodeOutcome(
                reference=InhibitReference(reason="init_effect_failed"),
                policy=params.default_policy(enabled=False),
                system_request=SystemRequestIntent.SAFE,
                faults=(fault,),
            ),
        )
    for result in applicable:
        edge = next(
            (
                edge
                for edge in EDGES
                if edge.source is state.node and edge.trigger is EdgeTrigger.EFFECT_COMPLETED
            ),
            None,
        )
        if edge is None:
            break
        if state.node is InitNode.HOME and not result.evidence_id:
            continue
        new_state = replace(state, node=edge.target)
        new_state, outcome = _node_step(new_state, inputs, params)
        outcome = replace(
            outcome,
            events=outcome.events
            + transition_event(
                GraphId.INIT,
                edge.source,
                edge.target,
                edge.trigger,
                inputs.timestamp_utc,
            ),
        )
        return new_state, GraphOutcome(node=edge.target, outcome=outcome, transition=edge)

    new_state, outcome = _node_step(state, inputs, params)
    return new_state, GraphOutcome(node=state.node, outcome=outcome)
