"""STOW graph: bounded move then confirmed hold (pure).

MOVING emits the bounded stow reference until control-path completion evidence
(`stow_complete` with fresh feedback and confirmed inhibit) commits the single
VERIFIED_STABLE edge to HELD. A pure elapsed timeout from `entered_s` latches
once: inhibit, GIMBAL_SAFETY_TIMEOUT, and one SAFE request.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

from dataclasses import replace

from flight.libs.types import FaultCode
from flight.payload.gimbal.request import InhibitReference
from flight.payload.graphs.base import (
    Edge,
    EdgeTrigger,
    GraphId,
    GraphOutcome,
    GraphSpec,
    NodeOutcome,
    SystemRequestIntent,
    TickInputs,
    transition_event,
)
from flight.payload.graphs.parameters import GraphParameters, encoder_fresh
from flight.payload.graphs.stow import held, moving
from flight.payload.graphs.stow.state import State, StowNode

EDGES: tuple[Edge[StowNode], ...] = (
    Edge(
        source=StowNode.MOVING,
        target=StowNode.HELD,
        trigger=EdgeTrigger.VERIFIED_STABLE,
    ),
)


def spec(params: GraphParameters) -> GraphSpec[StowNode]:
    """Declare the STOW graph.

    Inputs:
        params: Graph parameters supplying the default policy.

    Outputs:
        GraphSpec[StowNode]: MOVING to HELD on verified-stable evidence.
    """
    policy = params.default_policy(enabled=False)
    return GraphSpec(
        graph_id=GraphId.STOW,
        nodes=(StowNode.MOVING, StowNode.HELD),
        initial=StowNode.MOVING,
        edges=EDGES,
        imaging=policy.imaging,
        inference=policy.inference,
    )


def initial_state(inputs: TickInputs, params: GraphParameters) -> State:
    """Cold STOW state; MOVING entered at this tick's time.

    Inputs:
        inputs: First tick observations under this activation.
        params: Graph parameters.

    Outputs:
        State: MOVING with `entered_s` from the supplied monotonic time.
    """
    del params
    return State(
        activation_key=inputs.activation_key,
        node=StowNode.MOVING,
        entered_s=inputs.now_s,
    )


def _inhibit(
    state: State,
    params: GraphParameters,
    reason: str,
    system_request: SystemRequestIntent | None = None,
    faults: tuple[FaultCode, ...] = (),
) -> tuple[State, GraphOutcome[StowNode]]:
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


def step(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, GraphOutcome[StowNode]]:
    """Advance the STOW graph by one tick.

    Inputs:
        state: Current STOW state.
        inputs: Tick observations including stow completion evidence.
        params: Graph parameters.

    Outputs:
        tuple[State, GraphOutcome[StowNode]]: New state and the committed
            outcome; at most one edge commits per tick.
    """
    if inputs.activation_key != state.activation_key:
        return _inhibit(state, params, "activation_mismatch")
    if inputs.health.contained:
        return _inhibit(state, params, "contained", system_request=SystemRequestIntent.SAFE)
    if state.node is StowNode.MOVING:
        if state.timeout_latched:
            return _inhibit(state, params, "stow_timeout_latched")
        if inputs.now_s - state.entered_s >= params.config.gimbal.xeryon.stow_timeout_s:
            new_state = replace(state, timeout_latched=True)
            return new_state, GraphOutcome(
                node=state.node,
                outcome=NodeOutcome(
                    reference=InhibitReference(reason="stow_timeout"),
                    policy=params.default_policy(enabled=False),
                    system_request=SystemRequestIntent.SAFE,
                    faults=(FaultCode.GIMBAL_SAFETY_TIMEOUT,),
                ),
            )
        if not encoder_fresh(inputs, params):
            return _inhibit(state, params, "stale_feedback")
        if inputs.stow_complete and inputs.health.inhibit_confirmed:
            new_state = replace(state, node=StowNode.HELD)
            new_state, outcome = held.step(new_state, inputs, params)
            outcome = replace(
                outcome,
                events=outcome.events
                + transition_event(
                    GraphId.STOW,
                    StowNode.MOVING,
                    StowNode.HELD,
                    EdgeTrigger.VERIFIED_STABLE,
                    inputs.timestamp_utc,
                ),
            )
            return new_state, GraphOutcome(node=StowNode.HELD, outcome=outcome, transition=EDGES[0])
        new_state, outcome = moving.step(state, inputs, params)
        return new_state, GraphOutcome(node=StowNode.MOVING, outcome=outcome)
    new_state, outcome = held.step(state, inputs, params)
    return new_state, GraphOutcome(node=StowNode.HELD, outcome=outcome)
