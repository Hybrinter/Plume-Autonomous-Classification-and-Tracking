"""Tests for the pure graph runtime closed-union dispatch."""

import math

from flight.libs.messages import RoutedCommandMsg
from flight.libs.types import (
    ActivationKey,
    CommandId,
    Err,
    FaultCode,
    MessageType,
    Ok,
)
from flight.payload.graphs import idle, operate, runtime
from flight.payload.graphs.base import GraphId
from flight.payload.graphs.parameters import GraphParameters

from .support import TickBuilder


def _command(command_id: CommandId) -> RoutedCommandMsg:
    """Build one routed command envelope."""
    return RoutedCommandMsg(
        msg_type=MessageType.ROUTED_COMMAND,
        timestamp_utc="2026-06-01T00:00:00.000Z",
        target="payload",
        command_id=command_id.value,
        params={},
        source="ground",
        seq=1,
    )


def test_initial_state_per_graph(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Each GraphId yields its own graph's cold state."""
    inputs = tick(0.0, key)
    assert isinstance(runtime.initial_state(GraphId.IDLE, inputs, params), idle.State)
    assert runtime.initial_state(GraphId.SAFE, inputs, params).node.value == "inhibited"
    assert runtime.initial_state(GraphId.STOW, inputs, params).node.value == "moving"
    assert runtime.initial_state(GraphId.INIT, inputs, params).node.value == "selftest"
    assert isinstance(runtime.initial_state(GraphId.OPERATE, inputs, params), operate.State)


def test_step_dispatches_by_state_type(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """runtime.step matches on the concrete state class."""
    idle_state = runtime.initial_state(GraphId.IDLE, tick(0.0, key), params)
    state, outcome = runtime.step(idle_state, tick(0.02, key), params)
    assert isinstance(state, idle.State)
    assert outcome.node is idle.IdleNode.HOLD
    operate_state = runtime.initial_state(GraphId.OPERATE, tick(0.0, key), params)
    state2, outcome2 = runtime.step(operate_state, tick(0.02, key), params)
    assert isinstance(state2, operate.State)
    assert outcome2.node is operate.OperateNode.TRACKING


def test_apply_command_only_operate(
    params: GraphParameters, tick: TickBuilder, key: ActivationKey
) -> None:
    """Commands resolve only on OPERATE; other graphs reject."""
    idle_state = runtime.initial_state(GraphId.IDLE, tick(0.0, key), params)
    result = runtime.apply_command(
        idle_state, _command(CommandId.GIMBAL_HOLD), tick(0.02, key), params
    )
    assert isinstance(result, Err)
    assert result.error is FaultCode.COMMAND_INVALID
    operate_state = runtime.initial_state(GraphId.OPERATE, tick(0.0, key), params)
    result = runtime.apply_command(
        operate_state,
        _command(CommandId.GIMBAL_HOLD),
        tick(0.02, key, encoder_angle_rad=math.radians(20.0)),
        params,
    )
    assert isinstance(result, Ok)
    assert result.value.state.node is operate.OperateNode.HOLD
