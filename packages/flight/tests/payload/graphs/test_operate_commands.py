"""Tests for OPERATE apply_command: guards, edges, and hold conversions."""

import math
from dataclasses import replace

import pytest
from flight.libs.messages import RoutedCommandMsg
from flight.libs.types import CommandId, Err, FaultCode, MessageType, Ok
from flight.payload.gimbal.request import InhibitReference, PoseReference, RateReference
from flight.payload.graphs import operate
from flight.payload.graphs.base import SystemRequestIntent
from flight.payload.graphs.operate.state import State
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.records import ActivationKey, HealthSample

from .support import BlobBuilder, TickBuilder, VisionBuilder


def _command(
    command_id: CommandId,
    *,
    params: dict[str, str | int | float | bool] | None = None,
    target: str = "payload",
) -> RoutedCommandMsg:
    """Build one routed command envelope."""
    return RoutedCommandMsg(
        msg_type=MessageType.ROUTED_COMMAND,
        timestamp_utc="2026-06-01T00:00:00.000Z",
        target=target,
        command_id=command_id.value,
        params={} if params is None else params,
        source="ground",
        seq=3,
    )


def _tracking(params: GraphParameters, tick: TickBuilder, key: ActivationKey) -> State:
    """Cold TRACKING state."""
    return operate.initial_state(tick(0.0, key), params)


def test_hold_from_tracking_commits_manual_hold(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """GIMBAL_HOLD in TRACKING captures the fresh encoder pose."""
    angle = math.radians(20.0)
    state = _tracking(params, tick, key)
    result = operate.apply_command(
        state,
        _command(CommandId.GIMBAL_HOLD),
        tick(0.02, key, encoder_angle_rad=angle),
        params,
    )
    assert isinstance(result, Ok)
    new_state = result.value.state
    assert new_state.node is operate.OperateNode.HOLD
    assert new_state.hold.reason is operate.HoldReason.MANUAL
    assert new_state.hold.target_rad == angle
    outcome = result.value.outcome
    assert outcome.transition is not None
    assert outcome.transition.trigger.value == "command"
    assert isinstance(outcome.outcome.reference, PoseReference)


def test_hold_on_hunt_commits_to_hold(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """GIMBAL_HOLD is declared from both hunts to HOLD."""
    for node in (operate.OperateNode.REWIND, operate.OperateNode.FAST_REWIND):
        state = replace(
            _tracking(params, tick, key),
            node=node,
            loss_handled=True,
            rewind_entered_s=0.0,
        )
        result = operate.apply_command(
            state,
            _command(CommandId.GIMBAL_HOLD),
            tick(0.02, key, encoder_angle_rad=math.radians(20.0)),
            params,
        )
        assert isinstance(result, Ok)
        assert result.value.state.node is operate.OperateNode.HOLD


def test_home_and_goto_convert_hold_target(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """HOME and GOTO on HOLD re-aim the manual pose within bounds."""
    state = replace(
        _tracking(params, tick, key),
        node=operate.OperateNode.HOLD,
        hold=replace(
            _tracking(params, tick, key).hold,
            reason=operate.HoldReason.MANUAL,
            target_rad=math.radians(20.0),
        ),
    )
    home = operate.apply_command(state, _command(CommandId.GIMBAL_HOME), tick(0.02, key), params)
    assert isinstance(home, Ok)
    assert home.value.state.hold.target_rad == math.radians(params.config.gimbal.home_el_deg)
    goto = operate.apply_command(
        state,
        _command(CommandId.GIMBAL_GOTO, params={"el_deg": 30.0}),
        tick(0.02, key),
        params,
    )
    assert isinstance(goto, Ok)
    assert goto.value.state.hold.target_rad == math.radians(30.0)


def test_goto_outside_hardware_bounds_rejected(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """GOTO outside the hardware envelope is Err with state untouched."""
    state = replace(
        _tracking(params, tick, key),
        node=operate.OperateNode.HOLD,
        hold=replace(
            _tracking(params, tick, key).hold,
            reason=operate.HoldReason.MANUAL,
            target_rad=math.radians(20.0),
        ),
    )
    for el_deg in (
        params.config.gimbal.el_hw_min_deg - 1.0,
        params.config.gimbal.el_hw_max_deg + 1.0,
        math.inf,
    ):
        result = operate.apply_command(
            state,
            _command(CommandId.GIMBAL_GOTO, params={"el_deg": el_deg}),
            tick(0.02, key),
            params,
        )
        assert isinstance(result, Err)
        assert result.error is FaultCode.COMMAND_INVALID


def test_resume_cold_reenters_tracking(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """GIMBAL_RESUME from HOLD commits TRACKING with cold memory."""
    state = replace(
        _tracking(params, tick, key),
        node=operate.OperateNode.HOLD,
        tracked_blobs=(),
        hold=replace(
            _tracking(params, tick, key).hold,
            reason=operate.HoldReason.MANUAL,
            target_rad=math.radians(20.0),
        ),
    )
    result = operate.apply_command(
        state, _command(CommandId.GIMBAL_RESUME), tick(0.02, key), params
    )
    assert isinstance(result, Ok)
    new_state = result.value.state
    assert new_state.node is operate.OperateNode.TRACKING
    assert new_state.hold.target_rad is None
    assert new_state.residual.has_measurement is False
    assert isinstance(result.value.outcome.outcome.reference, RateReference)


def test_hold_command_wrong_node_rejected(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """HOME/GOTO/RESUME are declared only from HOLD; other nodes reject."""
    state = _tracking(params, tick, key)
    for command_id in (CommandId.GIMBAL_HOME, CommandId.GIMBAL_GOTO, CommandId.GIMBAL_RESUME):
        params_map: dict[str, str | int | float | bool] = (
            {"el_deg": 30.0} if command_id is CommandId.GIMBAL_GOTO else {}
        )
        result = operate.apply_command(
            state,
            _command(command_id, params=params_map),
            tick(0.02, key),
            params,
        )
        assert isinstance(result, Err)
        assert result.error is FaultCode.COMMAND_INVALID


def test_undeclared_command_rejected(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """A payload-target command with no declared edge is rejected."""
    state = _tracking(params, tick, key)
    result = operate.apply_command(state, _command(CommandId.GIMBAL_STOW), tick(0.02, key), params)
    assert isinstance(result, Err)
    assert result.error is FaultCode.COMMAND_INVALID


def test_wrong_target_and_bad_params_rejected(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Non-payload targets and schema-invalid params are rejected."""
    state = _tracking(params, tick, key)
    wrong_target = operate.apply_command(
        state,
        _command(CommandId.GIMBAL_HOLD, target="fault"),
        tick(0.02, key),
        params,
    )
    assert isinstance(wrong_target, Err)
    bad_params = operate.apply_command(
        state,
        _command(CommandId.GIMBAL_GOTO, params={"wrong": 1.0}),
        tick(0.02, key, encoder_angle_rad=math.radians(20.0)),
        params,
    )
    assert isinstance(bad_params, Err)


def test_stale_health_and_key_guards_reject(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Stale feedback, containment, and mismatched activation all reject."""
    state = _tracking(params, tick, key)
    stale = tick(
        0.02,
        key,
        health=HealthSample(feedback_valid=False, inhibit_confirmed=False, contained=False),
    )
    result = operate.apply_command(state, _command(CommandId.GIMBAL_HOLD), stale, params)
    assert isinstance(result, Err)
    contained = tick(
        0.02,
        key,
        health=HealthSample(feedback_valid=True, inhibit_confirmed=False, contained=True),
    )
    result = operate.apply_command(state, _command(CommandId.GIMBAL_HOLD), contained, params)
    assert isinstance(result, Err)
    wrong_key = tick(0.02, ActivationKey(epoch="test", sequence=9))
    result = operate.apply_command(state, _command(CommandId.GIMBAL_HOLD), wrong_key, params)
    assert isinstance(result, Err)


def test_command_commits_before_automatic_vision(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """A valid command edge wins over an automatic VISION_ACQUIRED edge."""
    state = replace(
        _tracking(params, tick, key),
        node=operate.OperateNode.REWIND,
        loss_handled=True,
        rewind_entered_s=0.0,
    )
    sample = vision(0.0, key, blobs=(blob(),), theta_g_rad=math.radians(20.0))
    inputs = tick(
        0.02,
        key,
        encoder_angle_rad=math.radians(20.0),
        vision=sample,
        command=_command(CommandId.GIMBAL_HOLD),
    )
    new_state, outcome = operate.step(state, inputs, params)
    assert new_state.node is operate.OperateNode.HOLD
    assert outcome.transition is not None
    assert outcome.transition.trigger.value == "command"


def _hold_state(params: GraphParameters, tick: TickBuilder, key: ActivationKey) -> State:
    """A MANUAL hold state under the current activation."""
    return replace(
        _tracking(params, tick, key),
        node=operate.OperateNode.HOLD,
        hold=replace(
            _tracking(params, tick, key).hold,
            reason=operate.HoldReason.MANUAL,
            target_rad=math.radians(20.0),
        ),
    )


@pytest.mark.parametrize(
    ("command_id", "params_map", "from_hold"),
    [
        (CommandId.GIMBAL_HOLD, None, False),
        (CommandId.GIMBAL_HOME, None, True),
        (CommandId.GIMBAL_GOTO, {"el_deg": 30.0}, True),
        (CommandId.GIMBAL_RESUME, None, True),
    ],
)
def test_flagged_vision_rejects_apply_command(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
    command_id: CommandId,
    params_map: dict[str, str | int | float | bool] | None,
    from_hold: bool,
) -> None:
    """An accepted flagged sample makes every command Err with state untouched."""
    angle = math.radians(20.0)
    state = _hold_state(params, tick, key) if from_hold else _tracking(params, tick, key)
    flagged = vision(0.02, key, blobs=(blob(),), theta_g_rad=angle, mode_flags=1)
    inputs = tick(
        0.04,
        key,
        encoder_angle_rad=angle,
        vision=flagged,
        command=_command(command_id, params=params_map),
    )
    result = operate.apply_command(state, _command(command_id, params=params_map), inputs, params)
    assert isinstance(result, Err)
    assert result.error is FaultCode.COMMAND_INVALID


def test_step_flagged_vision_inhibits_and_requests_safe(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """The graph step stays the fault/SAFE producer for flagged samples."""
    angle = math.radians(20.0)
    state = _tracking(params, tick, key)
    flagged = vision(0.02, key, blobs=(blob(),), theta_g_rad=angle, mode_flags=1)
    inputs = tick(
        0.04,
        key,
        encoder_angle_rad=angle,
        vision=flagged,
        command=_command(CommandId.GIMBAL_HOLD),
    )
    new_state, outcome = operate.step(state, inputs, params)
    assert new_state.node is operate.OperateNode.TRACKING
    assert outcome.transition is None
    assert isinstance(outcome.outcome.reference, InhibitReference)
    assert outcome.outcome.faults == (FaultCode.INFERENCE_NAN,)
    assert outcome.outcome.system_request is SystemRequestIntent.SAFE


def test_stale_context_flagged_sample_does_not_block_command(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """A flagged sample under another activation is ignored, not blocking."""
    angle = math.radians(20.0)
    state = _tracking(params, tick, key)
    stale = vision(
        0.02,
        ActivationKey(epoch="test", sequence=99),
        blobs=(blob(),),
        theta_g_rad=angle,
        mode_flags=1,
    )
    inputs = tick(0.04, key, encoder_angle_rad=angle, vision=stale)
    result = operate.apply_command(state, _command(CommandId.GIMBAL_HOLD), inputs, params)
    assert isinstance(result, Ok)


def test_resume_preserves_seen_vision_dedup(
    params: GraphParameters,
    tick: TickBuilder,
    vision: VisionBuilder,
    blob: BlobBuilder,
    key: ActivationKey,
) -> None:
    """RESUME keeps the activation-scoped seen window; a duplicate frame
    cannot re-establish liveness or feed the residual."""
    angle = math.radians(20.0)
    state = _hold_state(params, tick, key)
    frame = vision(0.02, key, frame_id="frame:dup", blobs=(blob(),), theta_g_rad=angle)
    state, _ = operate.step(state, tick(0.04, key, encoder_angle_rad=angle, vision=frame), params)
    assert any(frame_id == "frame:dup" for frame_id, _ in state.seen_vision)
    resumed = operate.apply_command(
        state, _command(CommandId.GIMBAL_RESUME), tick(0.06, key, encoder_angle_rad=angle), params
    )
    assert isinstance(resumed, Ok)
    new_state = resumed.value.state
    assert new_state.node is operate.OperateNode.TRACKING
    assert any(frame_id == "frame:dup" for frame_id, _ in new_state.seen_vision)
    duplicate = vision(0.06, key, frame_id="frame:dup", blobs=(blob(),), theta_g_rad=angle)
    final, _ = operate.step(
        new_state,
        tick(0.08, key, encoder_angle_rad=angle, vision=duplicate),
        params,
    )
    assert final.aggregate_live is False
    assert final.last_observation_s is None
    assert final.residual.has_measurement is False
    assert final.tracked_blobs == ()
