"""INIT HOME node: drive to the configured home pose, then wait (pure).

The node emits the HOME intent once and holds a PoseReference to the
configured home elevation under the pose envelope while feedback is fresh.
Without fresh feedback it inhibits and keeps waiting; arrival completion
requires a SUCCEEDED result with nonempty evidence.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

from dataclasses import replace

from flight.payload.gimbal.request import InhibitReference, PoseReference
from flight.payload.graphs.base import (
    EffectIntent,
    EffectKind,
    NodeOutcome,
    TickInputs,
)
from flight.payload.graphs.init.state import InitNode, State
from flight.payload.graphs.parameters import GraphParameters, encoder_fresh


def step(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, NodeOutcome[InitNode]]:
    """Emit the HOME intent once and pose to the home target when fresh.

    Inputs:
        state: Current INIT state.
        inputs: Tick observations.
        params: Graph parameters.

    Outputs:
        tuple[State, NodeOutcome[InitNode]]: State with the kind recorded in
            `issued`, plus the pose or inhibit reference, off policy, and the
            intent on its first emission.
    """
    policy = params.default_policy(enabled=False)
    if not encoder_fresh(inputs, params):
        return state, NodeOutcome(
            reference=InhibitReference(reason="init_home_stale_feedback"),
            policy=policy,
        )
    effects: tuple[EffectIntent, ...] = ()
    if EffectKind.HOME not in state.issued:
        state = replace(state, issued=state.issued | {EffectKind.HOME})
        effects = (
            EffectIntent(
                activation_key=state.activation_key,
                effect_id=EffectKind.HOME.value,
                kind=EffectKind.HOME,
            ),
        )
    return state, NodeOutcome(
        reference=PoseReference(target_rad=state.home_target_rad, envelope=params.pose_envelope),
        policy=policy,
        effects=effects,
    )
