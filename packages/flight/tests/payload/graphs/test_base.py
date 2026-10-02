"""Tests for the typed payload graph contracts in graphs/base.py."""

import math
from dataclasses import FrozenInstanceError
from enum import Enum

import pytest
from flight.libs.types import CommandId, Err, FaultCode, Ok
from flight.payload.graphs.base import (
    ActivationDisposition,
    ActivationKey,
    ActivationSnapshot,
    ActivationState,
    Edge,
    EdgeTrigger,
    GraphId,
    GraphSpec,
    ImagingOverride,
    ImagingPolicy,
    InferenceOverride,
    InferencePolicy,
    ModelSelection,
    PolicyLimits,
    accept_activation,
    command_target,
    resolve_policy,
    validate_policy,
    validate_spec,
)


class _Node(Enum):
    """Local test node vocabulary for spec validation."""

    A = "a"
    B = "b"
    C = "c"


def _imaging(
    *,
    acquisition_enabled: bool = True,
    capture_interval_s: float = 0.1,
    duty_cycle: float = 1.0,
    exposure_us: float = 1000.0,
    gain_db: float = 0.0,
) -> ImagingPolicy:
    """A valid default imaging policy."""
    return ImagingPolicy(
        acquisition_enabled=acquisition_enabled,
        capture_interval_s=capture_interval_s,
        duty_cycle=duty_cycle,
        exposure_us=exposure_us,
        gain_db=gain_db,
        publish_products=True,
    )


def _inference(enabled: bool = True, every_n_frames: int = 1) -> InferencePolicy:
    """A valid default inference policy."""
    return InferencePolicy(
        enabled=enabled,
        every_n_frames=every_n_frames,
        model=ModelSelection.CONFIGURED,
    )


def _limits() -> PolicyLimits:
    """Valid sensor limits."""
    return PolicyLimits(
        exposure_min_us=10.0,
        exposure_max_us=100_000.0,
        gain_min_db=0.0,
        gain_max_db=24.0,
        max_frame_rate_hz=30.0,
    )


def _spec(
    *,
    nodes: tuple[_Node, ...] = (_Node.A, _Node.B),
    initial: _Node = _Node.A,
    edges: tuple[Edge[_Node], ...] = (),
) -> GraphSpec[_Node]:
    """A spec helper with valid defaults."""
    return GraphSpec(
        graph_id=GraphId.OPERATE,
        nodes=nodes,
        initial=initial,
        edges=edges,
        imaging=_imaging(),
        inference=_inference(),
    )


def test_graph_id_serializes_lowercase() -> None:
    """GraphId values are lowercase telemetry names."""
    assert {g.value for g in GraphId} == {"idle", "stow", "safe", "init", "operate"}


def test_valid_spec_passes() -> None:
    """A declared initial node and valid edges validate."""
    spec = _spec(
        edges=(Edge(_Node.A, _Node.B, EdgeTrigger.VISION_ACQUIRED),),
    )
    assert isinstance(validate_spec(spec), Ok)


def test_empty_nodes_rejected() -> None:
    """A spec with no nodes is invalid."""
    assert isinstance(validate_spec(_spec(nodes=())), Err)


def test_duplicate_nodes_rejected() -> None:
    """Duplicate declared nodes are invalid."""
    spec = _spec(nodes=(_Node.A, _Node.A))
    assert isinstance(validate_spec(spec), Err)


def test_initial_must_be_declared() -> None:
    """An initial node outside the declared set is invalid."""
    spec = _spec(nodes=(_Node.A,), initial=_Node.B)
    result = validate_spec(spec)
    assert isinstance(result, Err)
    assert result.error is FaultCode.COMMAND_INVALID


def test_edge_endpoints_must_be_declared() -> None:
    """Edges with undeclared source or target are invalid."""
    bad_source = _spec(edges=(Edge(_Node.C, _Node.A, EdgeTrigger.TIMER_EXPIRED),))
    bad_target = _spec(edges=(Edge(_Node.A, _Node.C, EdgeTrigger.TIMER_EXPIRED),))
    assert isinstance(validate_spec(bad_source), Err)
    assert isinstance(validate_spec(bad_target), Err)


def test_duplicate_edges_rejected() -> None:
    """Two identical edges are invalid."""
    edge = Edge(_Node.A, _Node.B, EdgeTrigger.VISION_ACQUIRED)
    spec = _spec(edges=(edge, edge))
    assert isinstance(validate_spec(spec), Err)


def test_distinct_target_automatic_edges_allowed() -> None:
    """Automatic edges sharing (source, trigger) with distinct targets validate.

    Guard exclusivity belongs to the concrete graph; the spec only rejects
    exact duplicates.
    """
    spec = _spec(
        nodes=(_Node.A, _Node.B, _Node.C),
        edges=(
            Edge(_Node.A, _Node.B, EdgeTrigger.COAST_EXHAUSTED),
            Edge(_Node.A, _Node.C, EdgeTrigger.COAST_EXHAUSTED),
        ),
    )
    assert isinstance(validate_spec(spec), Ok)


def test_ambiguous_command_edges_rejected() -> None:
    """Two command edges sharing source and opcode with distinct targets fail."""
    spec = _spec(
        nodes=(_Node.A, _Node.B, _Node.C),
        edges=(
            Edge(_Node.A, _Node.B, EdgeTrigger.COMMAND, CommandId.GIMBAL_HOLD),
            Edge(_Node.A, _Node.C, EdgeTrigger.COMMAND, CommandId.GIMBAL_HOLD),
        ),
    )
    assert isinstance(validate_spec(spec), Err)


def test_command_edge_requires_command_id() -> None:
    """A COMMAND trigger without command_id is malformed."""
    spec = _spec(edges=(Edge(_Node.A, _Node.B, EdgeTrigger.COMMAND),))
    assert isinstance(validate_spec(spec), Err)


def test_noncommand_edge_rejects_command_id() -> None:
    """A non-command trigger carrying command_id is malformed."""
    spec = _spec(
        edges=(
            Edge(
                _Node.A,
                _Node.B,
                EdgeTrigger.TIMER_EXPIRED,
                CommandId.GIMBAL_HOLD,
            ),
        )
    )
    assert isinstance(validate_spec(spec), Err)


def _command_spec(*edges: Edge[_Node]) -> GraphSpec[_Node]:
    """Spec A->B HOLD self-loop vocabulary for command_target tests."""
    return _spec(nodes=(_Node.A, _Node.B), edges=edges)


def test_command_target_resolves_directed_edge() -> None:
    """A declared directed command edge resolves to its target."""
    spec = _command_spec(Edge(_Node.A, _Node.B, EdgeTrigger.COMMAND, CommandId.GIMBAL_HOLD))
    result = command_target(spec, _Node.A, CommandId.GIMBAL_HOLD)
    assert isinstance(result, Ok)
    assert result.value is _Node.B


def test_command_target_self_loop() -> None:
    """A command self-loop resolves to the same node."""
    spec = _command_spec(Edge(_Node.A, _Node.A, EdgeTrigger.COMMAND, CommandId.GIMBAL_RESUME))
    result = command_target(spec, _Node.A, CommandId.GIMBAL_RESUME)
    assert isinstance(result, Ok)
    assert result.value is _Node.A


def test_command_target_wrong_source_rejected() -> None:
    """A command declared only from another node does not apply."""
    spec = _command_spec(Edge(_Node.A, _Node.B, EdgeTrigger.COMMAND, CommandId.GIMBAL_HOLD))
    assert isinstance(command_target(spec, _Node.B, CommandId.GIMBAL_HOLD), Err)


def test_command_target_absent_command_rejected() -> None:
    """An undeclared opcode on a declared node is rejected."""
    spec = _command_spec(Edge(_Node.A, _Node.B, EdgeTrigger.COMMAND, CommandId.GIMBAL_HOLD))
    assert isinstance(command_target(spec, _Node.A, CommandId.GIMBAL_RESUME), Err)


def test_command_target_undeclared_node_rejected() -> None:
    """A node outside the declared set is rejected."""
    spec = _command_spec(Edge(_Node.A, _Node.B, EdgeTrigger.COMMAND, CommandId.GIMBAL_HOLD))
    assert isinstance(command_target(spec, _Node.C, CommandId.GIMBAL_HOLD), Err)


def test_command_target_invalid_spec_rejected() -> None:
    """An invalid spec fails before command resolution."""
    spec = _spec(nodes=(_Node.A, _Node.B), initial=_Node.C)
    assert isinstance(command_target(spec, _Node.A, CommandId.GIMBAL_HOLD), Err)


def test_valid_policy_passes() -> None:
    """The default imaging and inference policies validate."""
    assert isinstance(validate_policy(_imaging(), _inference(), _limits()), Ok)


def test_policy_rejects_nonfinite_fields() -> None:
    """NaN and infinite policy or limit values are invalid."""
    limits = _limits()
    for imaging in (
        _imaging(capture_interval_s=math.nan),
        _imaging(duty_cycle=math.inf),
        _imaging(exposure_us=math.inf),
        _imaging(gain_db=math.nan),
    ):
        assert isinstance(validate_policy(imaging, _inference(), limits), Err)
    bad_limits = PolicyLimits(10.0, math.inf, 0.0, 24.0, 30.0)
    assert isinstance(validate_policy(_imaging(), _inference(), bad_limits), Err)


def test_policy_rejects_unordered_limits() -> None:
    """Reversed or nonpositive limits are invalid."""
    bad = PolicyLimits(100.0, 10.0, 0.0, 24.0, 30.0)
    assert isinstance(validate_policy(_imaging(), _inference(), bad), Err)
    zero_fps = PolicyLimits(10.0, 100.0, 0.0, 24.0, 0.0)
    assert isinstance(validate_policy(_imaging(), _inference(), zero_fps), Err)


def test_policy_rejects_nonpositive_interval_and_bad_duty() -> None:
    """Zero interval and out-of-range duty are invalid."""
    limits = _limits()
    assert isinstance(validate_policy(_imaging(capture_interval_s=0.0), _inference(), limits), Err)
    assert isinstance(validate_policy(_imaging(duty_cycle=1.5), _inference(), limits), Err)
    assert isinstance(validate_policy(_imaging(duty_cycle=-0.1), _inference(), limits), Err)


def test_policy_rejects_bool_and_zero_decimation() -> None:
    """every_n_frames must be an exact positive int, not bool."""
    limits = _limits()
    assert isinstance(validate_policy(_imaging(), _inference(every_n_frames=True), limits), Err)
    assert isinstance(validate_policy(_imaging(), _inference(every_n_frames=0), limits), Err)
    assert isinstance(validate_policy(_imaging(), _inference(every_n_frames=-2), limits), Err)


def test_policy_rejects_inference_without_acquisition() -> None:
    """Enabled inference needs enabled acquisition and nonzero duty."""
    limits = _limits()
    off = validate_policy(_imaging(acquisition_enabled=False), _inference(enabled=True), limits)
    assert isinstance(off, Err)
    zero_duty = validate_policy(_imaging(duty_cycle=0.0), _inference(enabled=True), limits)
    assert isinstance(zero_duty, Err)
    disabled_inference = validate_policy(
        _imaging(acquisition_enabled=False), _inference(enabled=False), limits
    )
    assert isinstance(disabled_inference, Ok)


def test_policy_rejects_interval_below_camera_rate() -> None:
    """Capture interval must be at least one frame period."""
    limits = _limits()
    too_fast = validate_policy(_imaging(capture_interval_s=0.01), _inference(), limits)
    assert isinstance(too_fast, Err)
    at_limit = validate_policy(_imaging(capture_interval_s=1.0 / 30.0), _inference(), limits)
    assert isinstance(at_limit, Ok)


def test_policy_rejects_exposure_outside_bounds_or_interval() -> None:
    """Exposure must be inside limits and fit within the capture interval."""
    limits = _limits()
    too_low = validate_policy(_imaging(exposure_us=1.0), _inference(), limits)
    too_high = validate_policy(_imaging(exposure_us=200_000.0), _inference(), limits)
    exceeds_interval = validate_policy(
        _imaging(capture_interval_s=0.05, exposure_us=60_000.0), _inference(), limits
    )
    assert isinstance(too_low, Err)
    assert isinstance(too_high, Err)
    assert isinstance(exceeds_interval, Err)
    bad_gain = validate_policy(_imaging(gain_db=99.0), _inference(), limits)
    assert isinstance(bad_gain, Err)


def test_resolve_policy_inherits_unset_override_fields() -> None:
    """Overrides replace only their set fields; None fields inherit defaults."""
    resolved = resolve_policy(
        _imaging(),
        _inference(),
        ImagingOverride(exposure_us=2000.0),
        InferenceOverride(every_n_frames=3),
        _limits(),
    )
    assert isinstance(resolved, Ok)
    policy = resolved.value
    assert policy.imaging.exposure_us == 2000.0
    assert policy.imaging.gain_db == 0.0
    assert policy.imaging.acquisition_enabled is True
    assert policy.inference.every_n_frames == 3
    assert policy.inference.enabled is True
    assert policy.inference.model is ModelSelection.CONFIGURED


def test_resolve_policy_rejects_invalid_effective_policy() -> None:
    """An override producing an invalid complete policy is rejected."""
    resolved = resolve_policy(
        _imaging(acquisition_enabled=False),
        _inference(enabled=False),
        None,
        InferenceOverride(enabled=True),
        _limits(),
    )
    assert isinstance(resolved, Err)


def _snap(
    sequence: int,
    *,
    epoch: str = "epoch-1",
    graph_id: GraphId = GraphId.IDLE,
    reason: str = "boot",
) -> ActivationSnapshot:
    """Build an activation snapshot."""
    return ActivationSnapshot(
        key=ActivationKey(epoch=epoch, sequence=sequence),
        graph_id=graph_id,
        previous_graph=None,
        reason=reason,
    )


def test_first_activation_accepts() -> None:
    """A first snapshot with a valid epoch and sequence is accepted."""
    state = ActivationState(expected_epoch="epoch-1")
    decision = accept_activation(state, _snap(0))
    assert isinstance(decision, Ok)
    assert decision.value.disposition is ActivationDisposition.ACCEPTED
    assert decision.value.gap is False
    assert decision.value.state.last is not None


def test_unexpected_epoch_rejected() -> None:
    """Wrong or empty epochs are rejected without changing state."""
    state = ActivationState(expected_epoch="epoch-1")
    for epoch in ("epoch-2", ""):
        result = accept_activation(state, _snap(0, epoch=epoch))
        assert isinstance(result, Err)
        assert result.error is FaultCode.COMMAND_INVALID
        assert state.last is None


def test_negative_sequence_rejected() -> None:
    """A negative sequence is rejected."""
    state = ActivationState(expected_epoch="epoch-1")
    assert isinstance(accept_activation(state, _snap(-1)), Err)


def test_identical_key_is_duplicate() -> None:
    """The same key with identical contents is a duplicate, not re-entry."""
    state = ActivationState(expected_epoch="epoch-1", last=_snap(1))
    decision = accept_activation(state, _snap(1))
    assert isinstance(decision, Ok)
    assert decision.value.disposition is ActivationDisposition.DUPLICATE
    assert decision.value.state is state


def test_conflicting_key_rejected() -> None:
    """Same key with different contents is a synchronization error."""
    state = ActivationState(expected_epoch="epoch-1", last=_snap(1))
    conflict = _snap(1, graph_id=GraphId.SAFE, reason="fault")
    result = accept_activation(state, conflict)
    assert isinstance(result, Err)
    assert result.error is FaultCode.COMMAND_INVALID
    assert state.last is not conflict


def test_older_sequence_is_stale() -> None:
    """A lower sequence is ignored as stale."""
    state = ActivationState(expected_epoch="epoch-1", last=_snap(5))
    decision = accept_activation(state, _snap(3))
    assert isinstance(decision, Ok)
    assert decision.value.disposition is ActivationDisposition.STALE
    assert decision.value.state is state


def test_newer_sequence_accepts_and_reports_gap() -> None:
    """A newer sequence across a gap is accepted with gap True."""
    state = ActivationState(expected_epoch="epoch-1", last=_snap(1))
    contiguous = accept_activation(state, _snap(2))
    assert isinstance(contiguous, Ok)
    assert contiguous.value.gap is False
    gapped = accept_activation(state, _snap(4))
    assert isinstance(gapped, Ok)
    assert gapped.value.gap is True
    assert gapped.value.state.last is not None
    assert gapped.value.state.last.key.sequence == 4


def test_same_graph_newer_key_is_reentry() -> None:
    """A same-graph activation with a newer key is accepted as re-entry."""
    state = ActivationState(expected_epoch="epoch-1", last=_snap(1, graph_id=GraphId.IDLE))
    decision = accept_activation(state, _snap(2, graph_id=GraphId.IDLE))
    assert isinstance(decision, Ok)
    assert decision.value.disposition is ActivationDisposition.ACCEPTED
    assert decision.value.state.last is not None
    assert decision.value.state.last.graph_id is GraphId.IDLE


def test_activation_key_is_frozen() -> None:
    """ActivationKey rejects mutation."""
    key = ActivationKey(epoch="e", sequence=0)
    with pytest.raises(FrozenInstanceError):
        key.__setattr__("epoch", "x")
