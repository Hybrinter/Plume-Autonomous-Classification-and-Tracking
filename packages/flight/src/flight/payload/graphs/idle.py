"""IDLE graph: one hold node capturing a pose from fresh feedback (pure).

The single HOLD node captures the entry pose from the first fresh encoder
sample and holds it through the pose envelope. Without fresh feedback the node
inhibits and keeps no target; the first later fresh sample sets the target.
No imaging or inference runs in IDLE.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from flight.payload.gimbal.request import InhibitReference, PoseReference
from flight.payload.graphs.base import (
    GraphId,
    GraphOutcome,
    GraphSpec,
    NodeOutcome,
    SystemRequestIntent,
    TickInputs,
)
from flight.payload.graphs.parameters import GraphParameters, encoder_fresh
from flight.payload.records import ActivationKey


class IdleNode(Enum):
    """IDLE graph node vocabulary."""

    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class State:
    """IDLE graph state.

    Attributes:
        activation_key: Activation this state belongs to.
        node: Current node; always HOLD.
        target_rad: Captured hold elevation, rad, or None until fresh feedback.
    """

    activation_key: ActivationKey
    node: IdleNode
    target_rad: float | None = None


def spec(params: GraphParameters) -> GraphSpec[IdleNode]:
    """Declare the single-node IDLE graph.

    Inputs:
        params: Graph parameters supplying the default policy.

    Outputs:
        GraphSpec[IdleNode]: One HOLD node and no edges.
    """
    policy = params.default_policy(enabled=False)
    return GraphSpec(
        graph_id=GraphId.IDLE,
        nodes=(IdleNode.HOLD,),
        initial=IdleNode.HOLD,
        edges=(),
        imaging=policy.imaging,
        inference=policy.inference,
    )


def initial_state(inputs: TickInputs, params: GraphParameters) -> State:
    """Cold IDLE state; capture the entry pose when feedback is fresh.

    Inputs:
        inputs: First tick observations under this activation.
        params: Graph parameters.

    Outputs:
        State: HOLD with the encoder pose captured, or None while stale.
    """
    target = None
    if inputs.encoder is not None and encoder_fresh(inputs, params):
        target = inputs.encoder.angle_rad
    return State(
        activation_key=inputs.activation_key,
        node=IdleNode.HOLD,
        target_rad=target,
    )


def step(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, GraphOutcome[IdleNode]]:
    """Hold the captured pose; inhibit without fresh feedback.

    Inputs:
        state: Current IDLE state.
        inputs: Tick observations.
        params: Graph parameters.

    Outputs:
        tuple[State, GraphOutcome[IdleNode]]: Unchanged state except for a
            first target capture, plus the hold or inhibit outcome and off
            policy.
    """
    policy = params.default_policy(enabled=False)

    def inhibit(
        reason: str,
        system_request: SystemRequestIntent | None = None,
    ) -> tuple[State, GraphOutcome[IdleNode]]:
        return state, GraphOutcome(
            node=state.node,
            outcome=NodeOutcome(
                reference=InhibitReference(reason=reason),
                policy=policy,
                system_request=system_request,
            ),
        )

    if inputs.activation_key != state.activation_key:
        return inhibit("activation_mismatch")
    if inputs.health.contained:
        return inhibit("contained", system_request=SystemRequestIntent.SAFE)
    target = state.target_rad
    if target is None:
        if not encoder_fresh(inputs, params):
            return inhibit("idle_hold")
        encoder = inputs.encoder
        assert encoder is not None
        target = encoder.angle_rad
        state = replace(state, target_rad=target)
    elif not encoder_fresh(inputs, params):
        return inhibit("idle_stale_feedback")
    return state, GraphOutcome(
        node=state.node,
        outcome=NodeOutcome(
            reference=PoseReference(target_rad=target, envelope=params.pose_envelope),
            policy=policy,
        ),
    )
