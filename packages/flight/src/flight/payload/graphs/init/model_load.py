"""INIT MODEL_LOAD node: issue the model-load effect once, then wait (pure).

The node emits one MODEL_LOAD EffectIntent scoped to the activation and
inhibits motion while the shell loads and verifies the configured model pair.
Completion advances via the graph's EFFECT_COMPLETED edge.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

from dataclasses import replace

from flight.payload.gimbal.request import InhibitReference
from flight.payload.graphs.base import (
    EffectIntent,
    EffectKind,
    NodeOutcome,
    TickInputs,
)
from flight.payload.graphs.init.state import InitNode, State
from flight.payload.graphs.parameters import GraphParameters


def step(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, NodeOutcome[InitNode]]:
    """Emit the MODEL_LOAD intent once and inhibit.

    Inputs:
        state: Current INIT state.
        inputs: Tick observations.
        params: Graph parameters.

    Outputs:
        tuple[State, NodeOutcome[InitNode]]: State with the kind recorded in
            `issued`, plus the inhibit reference, off policy, and the intent
            on its first emission.
    """
    del inputs
    effects: tuple[EffectIntent, ...] = ()
    if EffectKind.MODEL_LOAD not in state.issued:
        state = replace(state, issued=state.issued | {EffectKind.MODEL_LOAD})
        effects = (
            EffectIntent(
                activation_key=state.activation_key,
                effect_id=EffectKind.MODEL_LOAD.value,
                kind=EffectKind.MODEL_LOAD,
            ),
        )
    return state, NodeOutcome(
        reference=InhibitReference(reason="init_model_load"),
        policy=params.default_policy(enabled=False),
        effects=effects,
    )
