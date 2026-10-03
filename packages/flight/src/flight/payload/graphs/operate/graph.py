"""OPERATE graph: tracking, hunts, and hold (pure).

TRACKING keeps the residual estimate; REWIND and FAST_REWIND hunt toward the
science limb; HOLD is limb-wait (auto-exits on vision) or manual (never).
Valid directed commands commit before automatic vision edges; at most one edge
commits per tick and the destination node's outcome applies same tick.
Stale or missing encoder feedback inhibits motion and keeps the enabled
imaging policy; other inhibit reasons disable acquisition and inference.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001, REQ-AIML-GIMB-008.
"""

from __future__ import annotations

import math
from dataclasses import replace

from flight.libs.commands import lookup_command, validate_command
from flight.libs.messages import RoutedCommandMsg
from flight.libs.types import CommandId, Err, FaultCode, Ok, Result
from flight.payload.gimbal.request import InhibitReference
from flight.payload.graphs.base import (
    CommandOutcome,
    Edge,
    EdgeTrigger,
    GraphId,
    GraphOutcome,
    GraphSpec,
    NodeOutcome,
    SystemRequestIntent,
    TickInputs,
    command_target,
    transition_event,
)
from flight.payload.graphs.operate import fast_rewind, hold, rewind, tracking
from flight.payload.graphs.operate.state import (
    HoldReason,
    HoldState,
    OperateNode,
    State,
    TargetState,
    accept_vision,
)
from flight.payload.graphs.parameters import GraphParameters, encoder_fresh
from flight.payload.records import CapturedVision

EDGES: tuple[Edge[OperateNode], ...] = (
    Edge(OperateNode.TRACKING, OperateNode.REWIND, EdgeTrigger.COAST_EXHAUSTED),
    Edge(OperateNode.TRACKING, OperateNode.HOLD, EdgeTrigger.COAST_EXHAUSTED),
    Edge(OperateNode.REWIND, OperateNode.FAST_REWIND, EdgeTrigger.TIMER_EXPIRED),
    Edge(OperateNode.REWIND, OperateNode.TRACKING, EdgeTrigger.VISION_ACQUIRED),
    Edge(OperateNode.FAST_REWIND, OperateNode.TRACKING, EdgeTrigger.VISION_ACQUIRED),
    Edge(OperateNode.REWIND, OperateNode.HOLD, EdgeTrigger.LIMB_ARRIVAL),
    Edge(OperateNode.FAST_REWIND, OperateNode.HOLD, EdgeTrigger.LIMB_ARRIVAL),
    Edge(OperateNode.HOLD, OperateNode.TRACKING, EdgeTrigger.VISION_ACQUIRED),
    Edge(OperateNode.TRACKING, OperateNode.HOLD, EdgeTrigger.COMMAND, CommandId.GIMBAL_HOLD),
    Edge(OperateNode.REWIND, OperateNode.HOLD, EdgeTrigger.COMMAND, CommandId.GIMBAL_HOLD),
    Edge(OperateNode.FAST_REWIND, OperateNode.HOLD, EdgeTrigger.COMMAND, CommandId.GIMBAL_HOLD),
    Edge(OperateNode.HOLD, OperateNode.HOLD, EdgeTrigger.COMMAND, CommandId.GIMBAL_HOLD),
    Edge(OperateNode.HOLD, OperateNode.HOLD, EdgeTrigger.COMMAND, CommandId.GIMBAL_HOME),
    Edge(OperateNode.HOLD, OperateNode.HOLD, EdgeTrigger.COMMAND, CommandId.GIMBAL_GOTO),
    Edge(OperateNode.HOLD, OperateNode.TRACKING, EdgeTrigger.COMMAND, CommandId.GIMBAL_RESUME),
)

_HUNTS = (OperateNode.REWIND, OperateNode.FAST_REWIND)


def spec(params: GraphParameters) -> GraphSpec[OperateNode]:
    """Declare the OPERATE graph.

    Inputs:
        params: Graph parameters supplying the default policy.

    Outputs:
        GraphSpec[OperateNode]: The four-node tracking graph with all edges.
    """
    policy = params.default_policy(enabled=True)
    return GraphSpec(
        graph_id=GraphId.OPERATE,
        nodes=(
            OperateNode.TRACKING,
            OperateNode.REWIND,
            OperateNode.FAST_REWIND,
            OperateNode.HOLD,
        ),
        initial=OperateNode.TRACKING,
        edges=EDGES,
        imaging=policy.imaging,
        inference=policy.inference,
    )


def initial_state(inputs: TickInputs, params: GraphParameters) -> State:
    """Cold TRACKING state with a residual history seeded from fresh feedback.

    Inputs:
        inputs: First tick observations under this activation.
        params: Graph parameters.

    Outputs:
        State: TRACKING, empty ancestry, residual seeded at the encoder when
            fresh, configured initial exposure.
    """
    filt = params.residual_filter
    if encoder_fresh(inputs, params):
        encoder = inputs.encoder
        assert encoder is not None
        history = filt.initial_history(
            t_s=encoder.t_s,
            encoder_angle_rad=encoder.angle_rad,
            encoder_endpoint_variance_rad2=encoder.angle_variance_rad2,
        )
    else:
        history = filt.initial_history()
    return State(
        activation_key=inputs.activation_key,
        node=OperateNode.TRACKING,
        tracked_blobs=(),
        aggregate_live=False,
        last_observation_s=None,
        miss_count=0,
        loss_handled=False,
        rewind_entered_s=None,
        residual=filt.initial_state(),
        residual_history=history,
        target=TargetState(
            r_cog_ecef_m=None,
            last_exposure_us=params.config.sensor.capture.initial_exposure_us,
            last_theta_los=0.0,
            last_omega_t_nom=0.0,
            last_omega_az_nom=0.0,
            last_omega_scene_el=0.0,
        ),
        hold=HoldState(reason=HoldReason.LIMB_WAIT, target_rad=None),
        seen_vision=(),
    )


def _inhibit(
    state: State,
    params: GraphParameters,
    reason: str,
    system_request: SystemRequestIntent | None = None,
    faults: tuple[FaultCode, ...] = (),
    *,
    enabled: bool = False,
) -> tuple[State, GraphOutcome[OperateNode]]:
    """Same-state inhibit outcome.

    ``enabled`` selects the imaging policy. Stale feedback passes true and
    keeps acquisition and inference. Other callers leave it false.

    Inputs:
        state: Current OPERATE state; returned unchanged.
        params: Graph parameters supplying the default policy.
        reason: Inhibit reference reason.
        system_request: Optional SAFE intent.
        faults: Fault codes raised with this inhibit.
        enabled: Whether acquisition and inference stay on.

    Outputs:
        tuple[State, GraphOutcome[OperateNode]]: Unchanged state and an
            inhibit outcome with no transition.
    """
    return state, GraphOutcome(
        node=state.node,
        outcome=NodeOutcome(
            reference=InhibitReference(reason=reason),
            policy=params.default_policy(enabled=enabled),
            system_request=system_request,
            faults=faults,
        ),
    )


def _node_step_to(
    node: OperateNode,
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, NodeOutcome[OperateNode]]:
    """Run the destination node's step while keeping the previous node visible."""
    match node:
        case OperateNode.TRACKING:
            return tracking.step(state, inputs, params)
        case OperateNode.REWIND:
            return rewind.step(state, inputs, params)
        case OperateNode.FAST_REWIND:
            return fast_rewind.step(state, inputs, params)
        case OperateNode.HOLD:
            return hold.step(state, inputs, params)


def _coast_exhausted(
    state: State, vision: CapturedVision | None, now_s: float, params: GraphParameters
) -> bool:
    """Bounded coast check: empty-sample release or observation-age release."""
    has_plume = vision is not None and len(vision.sample.blobs) > 0
    if has_plume or state.loss_handled:
        return False
    operate_cfg = params.config.controller.operate
    miss = state.miss_count + 1 if vision is not None else state.miss_count
    empty_release = vision is not None and miss >= operate_cfg.release_persistence_frames
    age_release = (
        state.last_observation_s is not None
        and max(0.0, now_s - state.last_observation_s) >= operate_cfg.max_observation_age_s
    )
    return empty_release or age_release


def _automatic_edge(
    state: State,
    vision: CapturedVision | None,
    inputs: TickInputs,
    params: GraphParameters,
) -> Edge[OperateNode] | None:
    """Evaluate automatic edge guards for the current node, in priority order.

    Accepted vision outranks limb arrival and the rewind timer; the coast
    exhaustion check is last and commits once per loss via loss_handled.
    """
    gimbal = params.config.gimbal
    operate_cfg = params.config.controller.operate
    has_plume = vision is not None and len(vision.sample.blobs) > 0
    encoder = inputs.encoder
    assert encoder is not None
    at_limb = (
        math.degrees(encoder.angle_rad) >= gimbal.el_science_max_deg - operate_cfg.limb_arrival_deg
    )
    node = state.node
    if has_plume:
        if node in _HUNTS or (
            node is OperateNode.HOLD and state.hold.reason is HoldReason.LIMB_WAIT
        ):
            return next(
                edge
                for edge in EDGES
                if edge.source is node and edge.trigger is EdgeTrigger.VISION_ACQUIRED
            )
        return None
    if node in _HUNTS:
        if at_limb:
            return next(
                edge
                for edge in EDGES
                if edge.source is node and edge.trigger is EdgeTrigger.LIMB_ARRIVAL
            )
        if node is OperateNode.REWIND:
            entered = state.rewind_entered_s
            if (
                entered is not None
                and inputs.now_s - entered >= params.config.controller.outer.rewind_sharp_max_s
            ):
                return next(
                    edge
                    for edge in EDGES
                    if edge.source is node and edge.trigger is EdgeTrigger.TIMER_EXPIRED
                )
        return None
    if node is OperateNode.TRACKING and _coast_exhausted(state, vision, inputs.now_s, params):
        trigger = EdgeTrigger.COAST_EXHAUSTED
        target = OperateNode.HOLD if at_limb else OperateNode.REWIND
        return next(
            edge
            for edge in EDGES
            if edge.source is node and edge.target is target and edge.trigger is trigger
        )
    return None


def _commit_transition(
    state: State,
    edge: Edge[OperateNode],
    inputs: TickInputs,
    params: GraphParameters,
) -> State:
    """Apply the bookkeeping deltas implied by a committed edge.

    COAST_EXHAUSTED latches loss_handled so the one release cannot re-fire.
    Limb-wait HOLD entries capture the fresh encoder angle. Hunt entries latch
    rewind_entered_s; TRACKING and hunt targets clear the hold target.
    """
    del params
    encoder = inputs.encoder
    state = replace(
        state,
        loss_handled=(True if edge.trigger is EdgeTrigger.COAST_EXHAUSTED else state.loss_handled),
        miss_count=(
            0
            if edge.trigger is EdgeTrigger.COAST_EXHAUSTED or edge.target is OperateNode.HOLD
            else state.miss_count
        ),
    )
    if edge.target is OperateNode.HOLD:
        reason = (
            HoldReason.LIMB_WAIT
            if edge.trigger in (EdgeTrigger.COAST_EXHAUSTED, EdgeTrigger.LIMB_ARRIVAL)
            else HoldReason.MANUAL
        )
        return replace(
            state,
            rewind_entered_s=None,
            hold=HoldState(
                reason=reason,
                target_rad=encoder.angle_rad if encoder is not None else None,
            ),
        )
    if edge.target in _HUNTS:
        return replace(
            state,
            rewind_entered_s=(inputs.now_s if state.node not in _HUNTS else state.rewind_entered_s),
            hold=HoldState(reason=HoldReason.LIMB_WAIT, target_rad=None),
        )
    return replace(
        state,
        rewind_entered_s=None,
        hold=HoldState(reason=HoldReason.LIMB_WAIT, target_rad=None),
    )


def _cold_tracking_state(state: State, inputs: TickInputs, params: GraphParameters) -> State:
    """Cold TRACKING fields for RESUME re-entry under the same activation."""
    filt = params.residual_filter
    encoder = inputs.encoder
    assert encoder is not None
    return replace(
        state,
        tracked_blobs=(),
        aggregate_live=False,
        last_observation_s=None,
        miss_count=0,
        loss_handled=False,
        rewind_entered_s=None,
        residual=filt.initial_state(),
        residual_history=filt.initial_history(
            t_s=encoder.t_s,
            encoder_angle_rad=encoder.angle_rad,
            encoder_endpoint_variance_rad2=encoder.angle_variance_rad2,
        ),
        target=TargetState(
            r_cog_ecef_m=None,
            last_exposure_us=params.config.sensor.capture.initial_exposure_us,
            last_theta_los=0.0,
            last_omega_t_nom=0.0,
            last_omega_az_nom=0.0,
            last_omega_scene_el=0.0,
        ),
        hold=HoldState(reason=HoldReason.LIMB_WAIT, target_rad=None),
    )


def _command_guards(
    state: State,
    command: RoutedCommandMsg,
    inputs: TickInputs,
    params: GraphParameters,
) -> Result[tuple[Edge[OperateNode], float | None], FaultCode]:
    """Validate a routed command and resolve the committed edge and hold target.

    Guard order: current activation, containment, fresh feedback, dictionary
    lookup and parameter validation, payload target, declared directed edge,
    and per-command bounds. Every failure is Err(COMMAND_INVALID) with the
    state untouched.
    """
    if inputs.activation_key != state.activation_key:
        return Err(FaultCode.COMMAND_INVALID)
    if inputs.health.contained:
        return Err(FaultCode.COMMAND_INVALID)
    if not encoder_fresh(inputs, params):
        return Err(FaultCode.COMMAND_INVALID)
    if command.target != "payload":
        return Err(FaultCode.COMMAND_INVALID)
    looked_up = lookup_command(command.command_id)
    if isinstance(looked_up, Err):
        return Err(looked_up.error)
    if isinstance(validate_command(looked_up.value, command.params), Err):
        return Err(FaultCode.COMMAND_INVALID)
    if looked_up.value.target != "payload":
        return Err(FaultCode.COMMAND_INVALID)
    try:
        command_id = CommandId(command.command_id)
    except ValueError:
        return Err(FaultCode.COMMAND_INVALID)
    found = command_target(spec(params), state.node, command_id)
    if isinstance(found, Err):
        return Err(FaultCode.COMMAND_INVALID)
    del found
    edge = next(
        edge
        for edge in EDGES
        if edge.source is state.node
        and edge.trigger is EdgeTrigger.COMMAND
        and edge.command_id is command_id
    )
    encoder = inputs.encoder
    assert encoder is not None
    gimbal = params.config.gimbal
    hold_target: float | None = None
    if command_id is CommandId.GIMBAL_HOLD:
        hold_target = encoder.angle_rad
    elif command_id is CommandId.GIMBAL_HOME:
        hold_target = math.radians(gimbal.home_el_deg)
    elif command_id is CommandId.GIMBAL_GOTO:
        el_deg = command.params["el_deg"]
        if not isinstance(el_deg, (int, float)) or isinstance(el_deg, bool):
            return Err(FaultCode.COMMAND_INVALID)
        el_deg = float(el_deg)
        if not math.isfinite(el_deg):
            return Err(FaultCode.COMMAND_INVALID)
        if not (gimbal.el_hw_min_deg <= el_deg <= gimbal.el_hw_max_deg):
            return Err(FaultCode.COMMAND_INVALID)
        hold_target = math.radians(el_deg)
    return Ok((edge, hold_target))


def apply_command(
    state: State,
    command: RoutedCommandMsg,
    inputs: TickInputs,
    params: GraphParameters,
) -> Result[CommandOutcome[State, OperateNode], FaultCode]:
    """Apply one routed command against declared command edges and guards.

    Inputs:
        state: Current OPERATE state.
        command: Routed command candidate for this tick.
        inputs: Tick observations.
        params: Graph parameters.

    Outputs:
        Result[CommandOutcome[State, OperateNode], FaultCode]: Ok with the new
            state and the destination node's outcome; Err(COMMAND_INVALID)
            with the state untouched on any guard failure.
    """
    guarded = _command_guards(state, command, inputs, params)
    if isinstance(guarded, Err):
        return Err(guarded.error)
    edge, hold_target = guarded.value
    vision = accept_vision(state, inputs, params)
    if vision is not None and vision.sample.mode_flags != 0:
        return Err(FaultCode.COMMAND_INVALID)
    eff_inputs = replace(inputs, vision=vision)
    if edge.target is OperateNode.HOLD:
        new_state = replace(
            state,
            hold=HoldState(reason=HoldReason.MANUAL, target_rad=hold_target),
            rewind_entered_s=None,
        )
    else:
        new_state = _cold_tracking_state(state, inputs, params)
    new_state, outcome = _node_step_to(edge.target, new_state, eff_inputs, params)
    outcome = replace(
        outcome,
        events=outcome.events
        + transition_event(
            GraphId.OPERATE,
            edge.source,
            edge.target,
            edge.trigger,
            inputs.timestamp_utc,
        ),
    )
    return Ok(
        CommandOutcome(
            state=new_state,
            outcome=GraphOutcome(node=edge.target, outcome=outcome, transition=edge),
        )
    )


def step(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, GraphOutcome[OperateNode]]:
    """Advance the OPERATE graph by one tick.

    Inputs:
        state: Current OPERATE state.
        inputs: Tick observations; an optional routed command candidate.
        params: Graph parameters.

    Outputs:
        tuple[State, GraphOutcome[OperateNode]]: New state and the committed
            outcome; a valid command commits before automatic edges, and at
            most one edge commits per tick.
    """
    if inputs.activation_key != state.activation_key:
        return _inhibit(state, params, "activation_mismatch")
    if inputs.health.contained:
        return _inhibit(state, params, "contained", system_request=SystemRequestIntent.SAFE)
    if not encoder_fresh(inputs, params):
        return _inhibit(state, params, "stale_feedback", enabled=True)

    vision = accept_vision(state, inputs, params)
    if vision is not None and vision.sample.mode_flags != 0:
        return _inhibit(
            state,
            params,
            "inference_flag",
            system_request=SystemRequestIntent.SAFE,
            faults=(FaultCode.INFERENCE_NAN,),
        )

    if inputs.command is not None:
        committed = apply_command(state, inputs.command, inputs, params)
        if isinstance(committed, Ok):
            return committed.value.state, committed.value.outcome

    eff_inputs = replace(inputs, vision=vision)
    edge = _automatic_edge(state, vision, inputs, params)
    if edge is not None:
        state = _commit_transition(state, edge, inputs, params)
        new_state, outcome = _node_step_to(edge.target, state, eff_inputs, params)
        if edge.trigger in (EdgeTrigger.COAST_EXHAUSTED, EdgeTrigger.LIMB_ARRIVAL):
            new_state = replace(new_state, miss_count=0)
        outcome = replace(
            outcome,
            events=outcome.events
            + transition_event(
                GraphId.OPERATE,
                edge.source,
                edge.target,
                edge.trigger,
                inputs.timestamp_utc,
            ),
        )
        return new_state, GraphOutcome(node=edge.target, outcome=outcome, transition=edge)
    new_state, outcome = _node_step_to(state.node, state, eff_inputs, params)
    return new_state, GraphOutcome(node=state.node, outcome=outcome)
