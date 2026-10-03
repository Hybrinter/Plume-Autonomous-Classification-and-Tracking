"""Payload runtime state: the threaded record owned by the shell's control path.

PayloadState bundles the mode-free servo memory, the activation-acceptance
state, the live graph state (None until the authority activates a mode), the
last committed control reference and effective policy, and shell revision
tokens that invalidate queued capture/effect work.

Pure value records; the imperative shell owns clocks, bus, and HAL.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from flight.payload.control import ServoState
from flight.payload.gimbal.request import ControlReference, InhibitReference
from flight.payload.graphs.base import ActivationState, EffectivePolicy, GraphId
from flight.payload.graphs.runtime import GraphState


@dataclass(frozen=True, slots=True)
class PayloadState:
    """The payload app's threaded state.

    Attributes:
        servo: Mode-free servo memory (encoder ring, PI, integrity, rate).
        activation: Activation acceptance state (expected epoch + last snapshot).
        graph: Live graph state, or None before the first accepted activation.
        reference: Last committed ControlReference for servo execution.
        policy: Last committed EffectivePolicy for capture gating.
        last_outer_s: Monotonic time of the last committed outer tick, or None.
        policy_revision: Bumped on every accepted activation/policy change; stale
            capture contexts carrying an older revision are discarded.
        control_revision: Bumped on every accepted activation; computed actuation
            from an older revision is obsolete.
    """

    servo: ServoState
    activation: ActivationState
    graph: GraphState | None
    reference: ControlReference
    policy: EffectivePolicy
    last_outer_s: float | None
    policy_revision: int = 0
    control_revision: int = 0


def graph_id_of(state: PayloadState) -> GraphId | None:
    """Return the active graph's GraphId, or None when unactivated."""
    if state.graph is None:
        return None
    if state.activation.last is None:
        return None
    return state.activation.last.graph_id


def node_of(state: PayloadState) -> Enum | None:
    """Return the active graph's current node enum, or None when unactivated."""
    if state.graph is None:
        return None
    node: Enum = state.graph.node
    return node


def graph_name_of(state: PayloadState) -> str:
    """Return the active graph's id value, or "" when unavailable."""
    graph_id = graph_id_of(state)
    return "" if graph_id is None else graph_id.value


def node_name_of(state: PayloadState) -> str:
    """Return the active node's value, or "" when unavailable."""
    node = node_of(state)
    return "" if node is None else str(node.value)


def unactivated_reference() -> ControlReference:
    """The boot reference before any activation: inhibit."""
    return InhibitReference("unactivated")
