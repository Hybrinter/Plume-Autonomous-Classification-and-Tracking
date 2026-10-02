"""INIT SELFTEST node: issue the self-test effect once, then wait (pure).

The node emits one SELFTEST EffectIntent scoped to the activation and inhibits
motion while the shell runs the bounded effect. Completion advances via the
graph's EFFECT_COMPLETED edge; this node issues no I/O itself.

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
    """Emit the SELFTEST intent once and inhibit.

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
    if EffectKind.SELFTEST not in state.issued:
        state = replace(state, issued=state.issued | {EffectKind.SELFTEST})
        effects = (
            EffectIntent(
                activation_key=state.activation_key,
                effect_id=EffectKind.SELFTEST.value,
                kind=EffectKind.SELFTEST,
            ),
        )
    return state, NodeOutcome(
        reference=InhibitReference(reason="init_selftest"),
        policy=params.default_policy(enabled=False),
        effects=effects,
    )
