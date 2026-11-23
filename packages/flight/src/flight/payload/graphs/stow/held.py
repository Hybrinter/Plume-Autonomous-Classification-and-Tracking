"""STOW HELD node: inhibit after confirmed stow arrival (pure).

HELD emits only an inhibit reference; the bounded move already completed and
the actuator stays inhibited until external graph selection.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

from flight.payload.gimbal.request import InhibitReference
from flight.payload.graphs.base import NodeOutcome, TickInputs
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.graphs.stow.state import State, StowNode


def step(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, NodeOutcome[StowNode]]:
    """Emit inhibit; the stow pose is already confirmed.

    Inputs:
        state: Current STOW state.
        inputs: Tick observations.
        params: Graph parameters.

    Outputs:
        tuple[State, NodeOutcome[StowNode]]: Same state and the inhibit
            reference with the off policy.
    """
    del inputs
    return state, NodeOutcome(
        reference=InhibitReference(reason="stow_held"),
        policy=params.default_policy(enabled=False),
    )
