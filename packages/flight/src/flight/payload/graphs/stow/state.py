"""STOW graph state and node vocabulary (pure).

MOVING drives the bounded stow reference until typed completion evidence or a
latched timeout. HELD emits inhibit once arrival is confirmed.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from flight.payload.records import ActivationKey


class StowNode(Enum):
    """STOW graph node vocabulary."""

    MOVING = "moving"
    HELD = "held"


@dataclass(frozen=True, slots=True)
class State:
    """STOW graph state.

    Attributes:
        activation_key: Activation this state belongs to.
        node: Current node.
        entered_s: Monotonic time the MOVING phase was entered.
        timeout_latched: True after the bounded stow window expired; the
            timeout fault and SAFE request emit exactly once.
    """

    activation_key: ActivationKey
    node: StowNode
    entered_s: float
    timeout_latched: bool = False
