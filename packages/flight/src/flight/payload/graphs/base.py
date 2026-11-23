"""Typed payload graph contracts: specs, edges, policies, inputs, and outcomes (pure).

A GraphSpec declares nodes and directed edges for one payload graph. Edges carry
an explicit trigger; command edges name a typed CommandId. validate_spec checks
topology only; validate_policy checks policy values against explicit limits;
command_target resolves exactly one directed command edge. Activation keys
order authority activations; accept_activation classifies each snapshot.

Pure: no I/O, no bus, no clock reads, no SystemMode. All value records are
frozen slots dataclasses. There is no graph engine, callable dispatch, or live
runtime here.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

from flight.libs.messages import RoutedCommandMsg, TelemetryEventMsg
from flight.libs.types import (
    ActivationKey,
    CommandId,
    Err,
    FaultCode,
    MessageType,
    Ok,
    Result,
)
from flight.payload.gimbal.request import ControlReference
from flight.payload.records import (
    CapturedVision,
    HealthSample,
    IssSample,
)
from flight.payload.tracking import EncoderSample, PredictorReferenceChange

__all__ = [
    "ActivationDecision",
    "ActivationDisposition",
    "ActivationKey",
    "ActivationSnapshot",
    "ActivationState",
    "CapturedVision",
    "CommandOutcome",
    "Edge",
    "EdgeTrigger",
    "EffectIntent",
    "EffectKind",
    "EffectResult",
    "EffectStatus",
    "EffectivePolicy",
    "GraphId",
    "GraphOutcome",
    "GraphSpec",
    "HealthSample",
    "ImagingOverride",
    "ImagingPolicy",
    "InferenceOverride",
    "InferencePolicy",
    "InitVerificationResult",
    "InitVerificationStatus",
    "IssSample",
    "ModelSelection",
    "NodeOutcome",
    "PolicyLimits",
    "SystemRequestIntent",
    "TickInputs",
    "accept_activation",
    "command_target",
    "resolve_policy",
    "transition_event",
    "validate_policy",
    "validate_spec",
]


class GraphId(Enum):
    """The five payload graphs. Lowercase values serialize for telemetry."""

    IDLE = "idle"
    STOW = "stow"
    SAFE = "safe"
    INIT = "init"
    OPERATE = "operate"


class EdgeTrigger(Enum):
    """Explicit trigger kinds for a directed graph edge."""

    VISION_ACQUIRED = "vision_acquired"
    COAST_EXHAUSTED = "coast_exhausted"
    TIMER_EXPIRED = "timer_expired"
    LIMB_ARRIVAL = "limb_arrival"
    COMMAND = "command"
    EFFECT_COMPLETED = "effect_completed"
    VERIFIED_STABLE = "verified_stable"


class ModelSelection(Enum):
    """Typed model-pair selection. CONFIGURED is the only selection for now."""

    CONFIGURED = "configured"


class EffectKind(Enum):
    """Typed INIT lifecycle effect kinds."""

    SELFTEST = "selftest"
    MODEL_LOAD = "model_load"
    HOME = "home"
    VERIFY_INIT = "verify_init"


class EffectStatus(Enum):
    """Completion status of one effect result."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class InitVerificationStatus(Enum):
    """Initialization verification outcome from the verifier seam."""

    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"


class SystemRequestIntent(Enum):
    """Payload-local request intents; the shell maps them onto bus requests."""

    SAFE = "safe"
    INIT_COMPLETE = "init_complete"


class ActivationDisposition(Enum):
    """Classification of one activation snapshot against the last key."""

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class Edge[NodeT: Enum]:
    """One directed edge in a graph spec.

    Attributes:
        source: Declared source node.
        target: Declared target node.
        trigger: Explicit trigger kind.
        command_id: Command opcode for COMMAND edges; None otherwise.
    """

    source: NodeT
    target: NodeT
    trigger: EdgeTrigger
    command_id: CommandId | None = None


@dataclass(frozen=True, slots=True)
class ImagingPolicy:
    """Camera acquisition policy owned by a graph or node.

    Attributes:
        acquisition_enabled: Whether the sensor streams.
        capture_interval_s: Seconds between capture opportunities.
        duty_cycle: Fraction of opportunities that capture, in [0, 1].
        exposure_us: Frame exposure, microseconds.
        gain_db: Sensor gain, dB.
        publish_products: Whether processed products are published.
    """

    acquisition_enabled: bool
    capture_interval_s: float
    duty_cycle: float
    exposure_us: float
    gain_db: float
    publish_products: bool


@dataclass(frozen=True, slots=True)
class InferencePolicy:
    """Inference cadence policy owned by a graph or node.

    Attributes:
        enabled: Whether inference runs on captured frames.
        every_n_frames: Run inference on every Nth accepted capture.
        model: Typed model-pair selection.
    """

    enabled: bool
    every_n_frames: int
    model: ModelSelection


@dataclass(frozen=True, slots=True)
class PolicyLimits:
    """Validated sensor bounds used to check policies and overrides.

    Attributes:
        exposure_min_us, exposure_max_us: Inclusive exposure bounds.
        gain_min_db, gain_max_db: Inclusive gain bounds.
        max_frame_rate_hz: Camera frame-rate ceiling for cadence checks.
    """

    exposure_min_us: float
    exposure_max_us: float
    gain_min_db: float
    gain_max_db: float
    max_frame_rate_hz: float


@dataclass(frozen=True, slots=True)
class EffectivePolicy:
    """Fully resolved imaging and inference policy for one node or graph."""

    imaging: ImagingPolicy
    inference: InferencePolicy


@dataclass(frozen=True, slots=True)
class ImagingOverride:
    """Optional per-node imaging fields; None inherits the graph default."""

    acquisition_enabled: bool | None = None
    capture_interval_s: float | None = None
    duty_cycle: float | None = None
    exposure_us: float | None = None
    gain_db: float | None = None
    publish_products: bool | None = None


@dataclass(frozen=True, slots=True)
class InferenceOverride:
    """Optional per-node inference fields; None inherits the graph default."""

    enabled: bool | None = None
    every_n_frames: int | None = None
    model: ModelSelection | None = None


@dataclass(frozen=True, slots=True)
class GraphSpec[NodeT: Enum]:
    """Declared topology and default policy for one payload graph.

    Attributes:
        graph_id: Graph identity.
        nodes: Declared node vocabulary; must contain the initial node.
        initial: Entry node.
        edges: Immutable directed edges.
        imaging: Default imaging policy for the graph.
        inference: Default inference policy for the graph.
    """

    graph_id: GraphId
    nodes: tuple[NodeT, ...]
    initial: NodeT
    edges: tuple[Edge[NodeT], ...]
    imaging: ImagingPolicy
    inference: InferencePolicy


@dataclass(frozen=True, slots=True)
class EffectIntent:
    """A graph's request for one bounded shell effect.

    Attributes:
        activation_key: Activation this effect belongs to.
        effect_id: Unique effect identity within the activation.
        kind: Typed effect kind.
    """

    activation_key: ActivationKey
    effect_id: str
    kind: EffectKind


@dataclass(frozen=True, slots=True)
class EffectResult:
    """Typed completion of one effect, scoped to its activation.

    Attributes:
        activation_key: Activation this result belongs to.
        effect_id: The intent's effect identity.
        kind: Typed effect kind.
        status: Completion status.
        fault: Fault code on failure; NONE otherwise.
        evidence_id: Opaque evidence identifier, or empty.
    """

    activation_key: ActivationKey
    effect_id: str
    kind: EffectKind
    status: EffectStatus
    fault: FaultCode = FaultCode.NONE
    evidence_id: str = ""


@dataclass(frozen=True, slots=True)
class InitVerificationResult:
    """Verifier-seam result scoped to the current activation.

    Attributes:
        activation_key: Activation this verification applies to.
        status: PENDING, VERIFIED, or FAILED.
        evidence_id: Opaque evidence identifier, or empty.
    """

    activation_key: ActivationKey
    status: InitVerificationStatus
    evidence_id: str = ""


@dataclass(frozen=True, slots=True)
class TickInputs:
    """Explicit observations for one graph tick; data only, no services.

    Attributes:
        now_s: Monotonic seconds of this tick.
        timestamp_utc: ISO stamp for emitted records; empty emits none.
        activation_key: Current activation under evaluation.
        encoder: Timestamped encoder observation, or None.
        navigation: ISS state, or None for unknown navigation.
        vision: Context-scoped vision capture, or None.
        health: Actuator health observation.
        command: Routed command candidate, or None.
        effect_results: Completed effect results for this tick.
        verification: Initialization verification result, or None.
        stow_complete: Bounded-stow completion evidence for this tick.
        reference_change: Explicit predictor-reference replacement; when
            supplied it takes precedence over the derived TRACKING CoG
            rebase, or None.
    """

    now_s: float
    timestamp_utc: str
    activation_key: ActivationKey
    encoder: EncoderSample | None
    navigation: IssSample | None
    vision: CapturedVision | None
    health: HealthSample
    command: RoutedCommandMsg | None = None
    effect_results: tuple[EffectResult, ...] = ()
    verification: InitVerificationResult | None = None
    stow_complete: bool = False
    reference_change: PredictorReferenceChange | None = None


@dataclass(frozen=True, slots=True)
class NodeOutcome[NodeT: Enum]:
    """Typed outputs of one node step.

    Attributes:
        reference: Motion reference for the servo path.
        policy: Resolved effective imaging and inference policy.
        transition: Declared target node on an intended transition, or None.
        effects: Bounded shell effect intents.
        system_request: Payload-local system-mode request intent, or None.
        faults: Fault codes raised this tick.
        events: Compact telemetry events.
    """

    reference: ControlReference
    policy: EffectivePolicy
    transition: NodeT | None = None
    effects: tuple[EffectIntent, ...] = ()
    system_request: SystemRequestIntent | None = None
    faults: tuple[FaultCode, ...] = ()
    events: tuple[TelemetryEventMsg, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphOutcome[NodeT: Enum]:
    """Outcome of one graph step or command application.

    Attributes:
        node: Active node after this evaluation.
        outcome: The node's typed outcome.
        transition: The committed edge when a transition occurred, or None.
    """

    node: NodeT
    outcome: NodeOutcome[NodeT]
    transition: Edge[NodeT] | None = None


@dataclass(frozen=True, slots=True)
class CommandOutcome[StateT, NodeT: Enum]:
    """Result of applying a validated command to graph state.

    Attributes:
        state: Graph state after the command.
        outcome: The graph outcome produced by the command.
    """

    state: StateT
    outcome: GraphOutcome[NodeT]


@dataclass(frozen=True, slots=True)
class ActivationSnapshot:
    """Semantic contents of one authoritative activation.

    Attributes:
        key: Activation identity (epoch, sequence).
        graph_id: Selected payload graph.
        previous_graph: Previously active graph, or None on initial activation.
        reason: Authority-provided reason.
        request_id: Correlated request identity, or None.
        recovery_authorized: Authority-approved recovery flag.
    """

    key: ActivationKey
    graph_id: GraphId
    previous_graph: GraphId | None
    reason: str
    request_id: str | None = None
    recovery_authorized: bool = False


@dataclass(frozen=True, slots=True)
class ActivationState:
    """Consumer-side activation bookkeeping.

    Attributes:
        expected_epoch: Composition-root-installed session epoch.
        last: Last accepted activation snapshot, or None before the first.
    """

    expected_epoch: str
    last: ActivationSnapshot | None = None


@dataclass(frozen=True, slots=True)
class ActivationDecision:
    """Classification of one snapshot and the resulting activation state.

    Attributes:
        state: Activation state after acceptance; unchanged on rejection.
        disposition: ACCEPTED, DUPLICATE, or STALE.
        gap: True when an accepted snapshot skipped sequence numbers.
    """

    state: ActivationState
    disposition: ActivationDisposition
    gap: bool = False


def validate_policy(
    imaging: ImagingPolicy,
    inference: InferencePolicy,
    limits: PolicyLimits,
) -> Result[None, FaultCode]:
    """Validate a complete imaging and inference policy against sensor limits.

    Inputs:
        imaging: Imaging policy values.
        inference: Inference policy values.
        limits: Validated sensor bounds and camera rate.

    Outputs:
        Result[None, FaultCode]: Ok(None) when valid; Err(COMMAND_INVALID)
            otherwise. All floats must be finite; limits must be ordered,
            finite, and positive where meaningful; capture interval must be
            positive and at least the frame-rate period; duty must lie in
            [0, 1]; exposure must be inside bounds and fit the interval;
            every_n_frames must be an exact positive int (bool rejected);
            enabled inference requires enabled acquisition and nonzero duty.
    """
    imaging_fields = (
        imaging.capture_interval_s,
        imaging.duty_cycle,
        imaging.exposure_us,
        imaging.gain_db,
    )
    if not all(math.isfinite(value) for value in imaging_fields):
        return Err(FaultCode.COMMAND_INVALID)
    limit_fields = (
        limits.exposure_min_us,
        limits.exposure_max_us,
        limits.gain_min_db,
        limits.gain_max_db,
        limits.max_frame_rate_hz,
    )
    if not all(math.isfinite(value) for value in limit_fields):
        return Err(FaultCode.COMMAND_INVALID)
    if not (
        0.0 < limits.exposure_min_us <= limits.exposure_max_us
        and limits.gain_min_db <= limits.gain_max_db
        and limits.max_frame_rate_hz > 0.0
    ):
        return Err(FaultCode.COMMAND_INVALID)
    if isinstance(inference.every_n_frames, bool) or not isinstance(inference.every_n_frames, int):
        return Err(FaultCode.COMMAND_INVALID)
    if inference.every_n_frames <= 0:
        return Err(FaultCode.COMMAND_INVALID)
    if not (0.0 <= imaging.duty_cycle <= 1.0):
        return Err(FaultCode.COMMAND_INVALID)
    if not (
        limits.exposure_min_us <= imaging.exposure_us <= limits.exposure_max_us
        and limits.gain_min_db <= imaging.gain_db <= limits.gain_max_db
    ):
        return Err(FaultCode.COMMAND_INVALID)
    if inference.enabled and not (imaging.acquisition_enabled and imaging.duty_cycle > 0.0):
        return Err(FaultCode.COMMAND_INVALID)
    if imaging.acquisition_enabled:
        if imaging.capture_interval_s <= 0.0:
            return Err(FaultCode.COMMAND_INVALID)
        if imaging.capture_interval_s + 1e-12 < 1.0 / limits.max_frame_rate_hz:
            return Err(FaultCode.COMMAND_INVALID)
        if imaging.exposure_us * 1.0e-6 > imaging.capture_interval_s + 1e-12:
            return Err(FaultCode.COMMAND_INVALID)
    elif imaging.capture_interval_s <= 0.0:
        return Err(FaultCode.COMMAND_INVALID)
    return Ok(None)


def resolve_policy(
    default_imaging: ImagingPolicy,
    default_inference: InferencePolicy,
    imaging_override: ImagingOverride | None,
    inference_override: InferenceOverride | None,
    limits: PolicyLimits,
) -> Result[EffectivePolicy, FaultCode]:
    """Resolve graph defaults plus node overrides into a validated policy.

    Inputs:
        default_imaging, default_inference: Graph-level defaults.
        imaging_override, inference_override: Optional per-node field overrides.
        limits: Validated sensor bounds.

    Outputs:
        Result[EffectivePolicy, FaultCode]: Ok with the complete effective
            policy; Err(COMMAND_INVALID) when the resolved policy is invalid.
    """
    imaging = default_imaging
    if imaging_override is not None:
        imaging = replace(
            imaging,
            acquisition_enabled=(
                default_imaging.acquisition_enabled
                if imaging_override.acquisition_enabled is None
                else imaging_override.acquisition_enabled
            ),
            capture_interval_s=(
                default_imaging.capture_interval_s
                if imaging_override.capture_interval_s is None
                else imaging_override.capture_interval_s
            ),
            duty_cycle=(
                default_imaging.duty_cycle
                if imaging_override.duty_cycle is None
                else imaging_override.duty_cycle
            ),
            exposure_us=(
                default_imaging.exposure_us
                if imaging_override.exposure_us is None
                else imaging_override.exposure_us
            ),
            gain_db=(
                default_imaging.gain_db
                if imaging_override.gain_db is None
                else imaging_override.gain_db
            ),
            publish_products=(
                default_imaging.publish_products
                if imaging_override.publish_products is None
                else imaging_override.publish_products
            ),
        )
    inference = default_inference
    if inference_override is not None:
        inference = replace(
            inference,
            enabled=(
                default_inference.enabled
                if inference_override.enabled is None
                else inference_override.enabled
            ),
            every_n_frames=(
                default_inference.every_n_frames
                if inference_override.every_n_frames is None
                else inference_override.every_n_frames
            ),
            model=(
                default_inference.model
                if inference_override.model is None
                else inference_override.model
            ),
        )
    valid = validate_policy(imaging, inference, limits)
    if isinstance(valid, Err):
        return valid
    return Ok(EffectivePolicy(imaging=imaging, inference=inference))


def validate_spec[NodeT: Enum](spec: GraphSpec[NodeT]) -> Result[None, FaultCode]:
    """Validate graph topology: nodes, initial, edges, and command edges.

    Inputs:
        spec: Declared graph spec.

    Outputs:
        Result[None, FaultCode]: Ok(None) when valid; Err(COMMAND_INVALID) for
            empty or duplicated nodes, an undeclared initial node, edges with
            undeclared endpoints, exact duplicate edges, command-trigger edges
            without a command_id, non-command edges carrying a command_id, or
            multiple command destinations sharing (source, command_id).
            Distinct-target automatic edges sharing (source, trigger) are
            allowed; the concrete graph owns guard exclusivity and selects the
            directed target edge itself.
    """
    if not spec.nodes:
        return Err(FaultCode.COMMAND_INVALID)
    if len(set(spec.nodes)) != len(spec.nodes):
        return Err(FaultCode.COMMAND_INVALID)
    if spec.initial not in spec.nodes:
        return Err(FaultCode.COMMAND_INVALID)
    seen_edges: set[Edge[NodeT]] = set()
    seen_commands: set[tuple[Enum, CommandId]] = set()
    for edge in spec.edges:
        if edge.source not in spec.nodes or edge.target not in spec.nodes:
            return Err(FaultCode.COMMAND_INVALID)
        if edge.trigger is EdgeTrigger.COMMAND:
            if edge.command_id is None:
                return Err(FaultCode.COMMAND_INVALID)
            command_identity = (edge.source, edge.command_id)
            if command_identity in seen_commands:
                return Err(FaultCode.COMMAND_INVALID)
            seen_commands.add(command_identity)
        elif edge.command_id is not None:
            return Err(FaultCode.COMMAND_INVALID)
        if edge in seen_edges:
            return Err(FaultCode.COMMAND_INVALID)
        seen_edges.add(edge)
    return Ok(None)


def command_target[NodeT: Enum](
    spec: GraphSpec[NodeT],
    node: NodeT,
    command_id: CommandId,
) -> Result[NodeT, FaultCode]:
    """Resolve the target of one directed command edge from a node.

    Inputs:
        spec: Declared graph spec (validated first).
        node: Declared source node.
        command_id: Command opcode to match.

    Outputs:
        Result[NodeT, FaultCode]: Ok with the edge target; Err(COMMAND_INVALID)
            when the spec is invalid, the node is undeclared, or zero/multiple
            command edges match. Parameter and health guards belong to the
            graph, not here.
    """
    valid = validate_spec(spec)
    if isinstance(valid, Err):
        return valid
    if node not in spec.nodes:
        return Err(FaultCode.COMMAND_INVALID)
    matches = [
        edge
        for edge in spec.edges
        if edge.trigger is EdgeTrigger.COMMAND
        and edge.source == node
        and edge.command_id is command_id
    ]
    if len(matches) != 1:
        return Err(FaultCode.COMMAND_INVALID)
    return Ok(matches[0].target)


def transition_event(
    graph_id: GraphId,
    source: Enum,
    target: Enum,
    trigger: EdgeTrigger,
    timestamp_utc: str,
) -> tuple[TelemetryEventMsg, ...]:
    """Build the compact node-transition telemetry event for one commit.

    Inputs:
        graph_id: Owning graph.
        source, target: Committed edge endpoints.
        trigger: Trigger that committed the edge.
        timestamp_utc: ISO stamp; empty emits no event.

    Outputs:
        tuple[TelemetryEventMsg, ...]: One node_transition event, or empty.
    """
    if not timestamp_utc:
        return ()
    return (
        TelemetryEventMsg(
            msg_type=MessageType.TELEMETRY_EVENT,
            timestamp_utc=timestamp_utc,
            subsystem="payload",
            event_name="node_transition",
            payload={
                "graph": graph_id.value,
                "from": source.value,
                "to": target.value,
                "reason": trigger.value,
            },
        ),
    )


def accept_activation(
    state: ActivationState,
    snapshot: ActivationSnapshot,
) -> Result[ActivationDecision, FaultCode]:
    """Classify one activation snapshot against the last accepted key.

    Inputs:
        state: Consumer activation state (expected epoch, last snapshot).
        snapshot: Incoming activation snapshot.

    Outputs:
        Result[ActivationDecision, FaultCode]: Ok with an updated state and a
            disposition; Err(COMMAND_INVALID) for an empty or unexpected epoch,
            a negative sequence, or conflicting contents under an existing key.
            On Err the input state is unchanged. A first snapshot accepts any
            nonnegative sequence. Equal key and identical contents is
            DUPLICATE; lower sequence is STALE; newer is ACCEPTED, with gap
            True when the sequence skipped.
    """
    key = snapshot.key
    if not key.epoch or key.epoch != state.expected_epoch or key.sequence < 0:
        return Err(FaultCode.COMMAND_INVALID)
    last = state.last
    if last is None:
        return Ok(
            ActivationDecision(
                state=replace(state, last=snapshot),
                disposition=ActivationDisposition.ACCEPTED,
            )
        )
    if key == last.key:
        if snapshot == last:
            return Ok(ActivationDecision(state=state, disposition=ActivationDisposition.DUPLICATE))
        return Err(FaultCode.COMMAND_INVALID)
    if key.sequence < last.key.sequence:
        return Ok(ActivationDecision(state=state, disposition=ActivationDisposition.STALE))
    gap = key.sequence > last.key.sequence + 1
    return Ok(
        ActivationDecision(
            state=replace(state, last=snapshot),
            disposition=ActivationDisposition.ACCEPTED,
            gap=gap,
        )
    )
