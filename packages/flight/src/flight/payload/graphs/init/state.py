"""INIT graph state and node vocabulary (pure).

The graph walks SELFTEST -> MODEL_LOAD -> HOME -> READY on activation-scoped
effect completions, then waits for verification before requesting IDLE. State
records which effect intents were issued so stale or unsolicited results never
advance progress.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from flight.payload.graphs.base import EffectKind
from flight.payload.records import ActivationKey


class InitNode(Enum):
    """INIT graph node vocabulary."""

    SELFTEST = "selftest"
    MODEL_LOAD = "model_load"
    HOME = "home"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class State:
    """INIT graph state.

    Attributes:
        activation_key: Activation this state belongs to.
        node: Current node.
        issued: Effect kinds whose intents were already emitted; each kind is
            issued at most once per activation.
        failed: True after a failed effect or verification; latches inhibit
            and a single SAFE request.
        requested_idle: True after the one INIT_COMPLETE request emitted on a
            VERIFIED ready result; prevents request flooding.
        home_target_rad: Configured home pose, rad, captured at entry.
    """

    activation_key: ActivationKey
    node: InitNode
    issued: frozenset[EffectKind]
    failed: bool
    requested_idle: bool
    home_target_rad: float
