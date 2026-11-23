"""OPERATE HOLD node: captured pose hold at the limb or manual target (pure).

HOLD keeps a PoseReference to a captured or commanded elevation under the pose
envelope. Limb-wait entries capture the fresh encoder angle on arrival; the
residual is not fed while holding. Without a target and without fresh feedback
the node inhibits until feedback establishes one.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

from dataclasses import replace

from flight.libs.types import Err, FaultCode
from flight.payload.gimbal.request import InhibitReference, PoseReference
from flight.payload.graphs.base import NodeOutcome, SystemRequestIntent, TickInputs
from flight.payload.graphs.operate.state import OperateNode, State, bookkeep_vision
from flight.payload.graphs.parameters import GraphParameters, encoder_fresh


def step(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, NodeOutcome[OperateNode]]:
    """One HOLD tick: hold the captured/commanded pose or inhibit.

    Inputs:
        state: OPERATE state; `hold` carries the reason and target.
        inputs: Tick observations with the resolved effective vision sample.
        params: Graph parameters.

    Outputs:
        tuple[State, NodeOutcome[OperateNode]]: Updated state and the pose or
            inhibit reference under the enabled policy.
    """
    resolved = params.operating_policy(params.config.payload_policy.hold)
    if isinstance(resolved, Err):
        return state, NodeOutcome(
            reference=InhibitReference(reason="invalid_policy"),
            policy=params.default_policy(enabled=False),
            system_request=SystemRequestIntent.SAFE,
            faults=(FaultCode.COMMAND_INVALID,),
        )
    policy = resolved.value
    state = bookkeep_vision(state, inputs.vision, inputs, params)
    state = replace(state, aggregate_live=False, last_rate_decision=None)
    target = state.hold.target_rad
    if target is None:
        if not encoder_fresh(inputs, params):
            return state, NodeOutcome(
                reference=InhibitReference(reason="hold_stale_feedback"), policy=policy
            )
        encoder = inputs.encoder
        assert encoder is not None
        target = encoder.angle_rad
        state = replace(state, hold=replace(state.hold, target_rad=target))
    elif not encoder_fresh(inputs, params):
        return state, NodeOutcome(
            reference=InhibitReference(reason="hold_stale_feedback"), policy=policy
        )
    state = replace(state, node=OperateNode.HOLD)
    return state, NodeOutcome(
        reference=PoseReference(target_rad=target, envelope=params.pose_envelope),
        policy=policy,
    )
