"""STOW MOVING node: bounded move to the stow pose (pure).

The node emits the configured stow target under the hardware envelope capped
at the bounded stow reference rate, with the configured timeout. Completion
and timeout handling live in the graph; this file only produces the moving
reference.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

import math

from flight.payload.gimbal.request import StowReference
from flight.payload.graphs.base import NodeOutcome, TickInputs
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.graphs.stow.state import State, StowNode


def step(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, NodeOutcome[StowNode]]:
    """Emit the bounded stow reference.

    Inputs:
        state: Current STOW state.
        inputs: Tick observations.
        params: Graph parameters.

    Outputs:
        tuple[State, NodeOutcome[StowNode]]: Same state and the stow
            reference with the off policy.
    """
    del inputs
    reference = StowReference(
        target_rad=math.radians(params.config.gimbal.stow_el_deg),
        envelope=params.stow_envelope,
        timeout_s=params.config.gimbal.xeryon.stow_timeout_s,
    )
    return state, NodeOutcome(reference=reference, policy=params.default_policy(enabled=False))
