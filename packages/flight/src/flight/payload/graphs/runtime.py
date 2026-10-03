"""Payload graph runtime: closed-union dispatch across the five graphs (pure).

The runtime is not an engine: `step` and `apply_command` are explicit `match`
dispatch over the closed graph-state union, and `initial_state` builds the
cold graph state selected by the caller's authoritative activation. No
activation bookkeeping, epoch generation, or graph selection lives here; the
runtime caller (shell wiring in the message cutover) owns those choices.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001, REQ-AIML-GIMB-008.
"""

from __future__ import annotations

from flight.libs.messages import RoutedCommandMsg
from flight.libs.types import Err, FaultCode, Result
from flight.payload.graphs import idle, init, operate, safe, stow
from flight.payload.graphs.base import (
    CommandOutcome,
    GraphId,
    GraphOutcome,
    GraphSpec,
    TickInputs,
)
from flight.payload.graphs.parameters import GraphParameters

GraphState = idle.State | safe.State | stow.State | init.State | operate.State
"""Closed union of every concrete graph state."""

GraphOutcomeUnion = (
    GraphOutcome[idle.IdleNode]
    | GraphOutcome[safe.SafeNode]
    | GraphOutcome[stow.StowNode]
    | GraphOutcome[init.InitNode]
    | GraphOutcome[operate.OperateNode]
)
"""Closed union of every concrete graph outcome."""

CommandOutcomeUnion = CommandOutcome[operate.State, operate.OperateNode]

GraphSpecUnion = (
    GraphSpec[idle.IdleNode]
    | GraphSpec[safe.SafeNode]
    | GraphSpec[stow.StowNode]
    | GraphSpec[init.InitNode]
    | GraphSpec[operate.OperateNode]
)
"""Closed union of every concrete graph spec."""


def initial_state(
    graph_id: GraphId,
    inputs: TickInputs,
    params: GraphParameters,
) -> GraphState:
    """Build the cold state for the caller's authoritative graph selection.

    Inputs:
        graph_id: The graph selected by the external authority.
        inputs: First tick observations under this activation.
        params: Graph parameters.

    Outputs:
        GraphState: Cold state for the selected graph.
    """
    match graph_id:
        case GraphId.IDLE:
            return idle.initial_state(inputs, params)
        case GraphId.SAFE:
            return safe.initial_state(inputs, params)
        case GraphId.STOW:
            return stow.initial_state(inputs, params)
        case GraphId.INIT:
            return init.initial_state(inputs, params)
        case GraphId.OPERATE:
            return operate.initial_state(inputs, params)


def spec(
    graph_id: GraphId,
    params: GraphParameters,
) -> GraphSpecUnion:
    """Return the declared spec (topology + default policy) for one graph."""
    match graph_id:
        case GraphId.IDLE:
            return idle.spec(params)
        case GraphId.SAFE:
            return safe.spec(params)
        case GraphId.STOW:
            return stow.spec(params)
        case GraphId.INIT:
            return init.spec(params)
        case GraphId.OPERATE:
            return operate.spec(params)


def step(
    state: GraphState,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[GraphState, GraphOutcomeUnion]:
    """Advance whichever graph owns `state` by one tick.

    Inputs:
        state: Current state of one of the five graphs.
        inputs: Tick observations.
        params: Graph parameters.

    Outputs:
        tuple[GraphState, GraphOutcomeUnion]: The new state and outcome.
    """
    match state:
        case idle.State():
            return idle.step(state, inputs, params)
        case safe.State():
            return safe.step(state, inputs, params)
        case stow.State():
            return stow.step(state, inputs, params)
        case init.State():
            return init.step(state, inputs, params)
        case operate.State():
            return operate.step(state, inputs, params)


def apply_command(
    state: GraphState,
    command: RoutedCommandMsg,
    inputs: TickInputs,
    params: GraphParameters,
) -> Result[CommandOutcomeUnion, FaultCode]:
    """Apply a routed command; only OPERATE declares command edges.

    Inputs:
        state: Current state of one of the five graphs.
        command: Routed command candidate.
        inputs: Tick observations.
        params: Graph parameters.

    Outputs:
        Result[CommandOutcomeUnion, FaultCode]: OPERATE's command outcome, or
            Err(COMMAND_INVALID) for every other graph.
    """
    match state:
        case operate.State():
            return operate.apply_command(state, command, inputs, params)
        case _:
            return Err(FaultCode.COMMAND_INVALID)
