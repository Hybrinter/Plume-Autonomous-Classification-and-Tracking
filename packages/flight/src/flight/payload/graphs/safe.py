"""SAFE graph: one inhibited node with no motion references (pure).

INHIBITED always emits an InhibitReference regardless of feedback state and
runs no imaging or inference. SAFE is independent motion inhibition: the graph
holds no pose, issues no stow, and waits for external graph selection.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from flight.libs.types import ActivationKey
from flight.payload.gimbal.request import InhibitReference
from flight.payload.graphs.base import (
    GraphId,
    GraphOutcome,
    GraphSpec,
    NodeOutcome,
    TickInputs,
)
from flight.payload.graphs.parameters import GraphParameters

_INHIBIT = InhibitReference(reason="safe")


class SafeNode(Enum):
    """SAFE graph node vocabulary."""

    INHIBITED = "inhibited"


@dataclass(frozen=True, slots=True)
class State:
    """SAFE graph state.

    Attributes:
        activation_key: Activation this state belongs to.
        node: Current node; always INHIBITED.
    """

    activation_key: ActivationKey
    node: SafeNode


def spec(params: GraphParameters) -> GraphSpec[SafeNode]:
    """Declare the single-node SAFE graph.

    Inputs:
        params: Graph parameters supplying the default policy.

    Outputs:
        GraphSpec[SafeNode]: One INHIBITED node and no edges.
    """
    policy = params.default_policy(enabled=False)
    return GraphSpec(
        graph_id=GraphId.SAFE,
        nodes=(SafeNode.INHIBITED,),
        initial=SafeNode.INHIBITED,
        edges=(),
        imaging=policy.imaging,
        inference=policy.inference,
    )


def initial_state(inputs: TickInputs, params: GraphParameters) -> State:
    """Cold SAFE state.

    Inputs:
        inputs: First tick observations under this activation.
        params: Graph parameters.

    Outputs:
        State: INHIBITED under this activation.
    """
    del params
    return State(activation_key=inputs.activation_key, node=SafeNode.INHIBITED)


def step(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, GraphOutcome[SafeNode]]:
    """Emit inhibit regardless of feedback; SAFE never poses or tracks.

    Inputs:
        state: Current SAFE state.
        inputs: Tick observations.
        params: Graph parameters.

    Outputs:
        tuple[State, GraphOutcome[SafeNode]]: Same state and the inhibit
            outcome with the off policy.
    """
    del inputs
    return state, GraphOutcome(
        node=state.node,
        outcome=NodeOutcome(reference=_INHIBIT, policy=params.default_policy(enabled=False)),
    )
