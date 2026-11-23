"""Tests for the analysis signal registry (datapoints)."""

import math
from dataclasses import replace

import pytest
from flight.libs.config import PactConfig
from flight.libs.time import ManualClock
from flight.libs.types import ActivationKey
from flight.payload.graphs import runtime
from flight.payload.graphs.base import ActivationSnapshot, GraphId, TickInputs
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.records import HealthSample
from flight.payload.tracking import EncoderSample
from sim.scene import build_frames, plume_detector
from sim.sil import build_sil_system
from tools.analysis.datapoints import (
    GROUPS,
    MESSAGE_TYPES,
    REGISTRY,
    SampleContext,
    Signal,
    SignalKind,
    accumulable_names,
    is_event_rate,
    signal_names,
    signals_for_group,
)
from tools.analysis.recorder import sample_devices

_EXPECTED_GROUPS = {
    "system",
    "bus",
    "payload",
    "fault",
    "iss_iface",
    "thermal",
    "electrical",
    "command_router",
    "storage",
    "downlink",
    "model_deploy",
}


def test_registry_is_nonempty_and_uniquely_named() -> None:
    """The registry has many signals and no duplicate names."""
    assert len(REGISTRY) > 200
    names = [signal.name for signal in REGISTRY]
    assert len(names) == len(set(names))


def test_registry_covers_every_app_and_the_bus() -> None:
    """Every flight app group, the bus, and the system rollup are represented."""
    assert _EXPECTED_GROUPS.issubset(set(GROUPS))
    for group in _EXPECTED_GROUPS:
        assert len(signals_for_group(group)) >= 1


def test_signal_fields_are_wellformed() -> None:
    """Each signal has a dotted name, a known group, a kind, and a callable extractor."""
    for signal in REGISTRY:
        assert isinstance(signal, Signal)
        assert signal.name and signal.group and signal.title
        assert signal.group in _EXPECTED_GROUPS
        assert isinstance(signal.kind, SignalKind)
        assert callable(signal.extract)


def test_both_kinds_present() -> None:
    """The registry has both numeric and categorical signals."""
    kinds = {signal.kind for signal in REGISTRY}
    assert kinds == {SignalKind.NUMERIC, SignalKind.CATEGORICAL}


def test_bus_family_has_one_signal_per_message_type() -> None:
    """The bus group exposes a publish-count signal for every one of the 19 message types."""
    bus_names = {signal.name for signal in signals_for_group("bus")}
    for message_type in MESSAGE_TYPES:
        short = message_type.__name__.removesuffix("Msg")
        assert f"bus.published.{short}" in bus_names


def test_accumulable_names_are_event_rate_registry_signals() -> None:
    """Every accumulable name is a registered per-step event-rate numeric signal."""
    registry_by_name = {signal.name: signal for signal in REGISTRY}
    accumulable = accumulable_names()
    assert accumulable  # non-empty
    for name in accumulable:
        signal = registry_by_name[name]
        assert signal.kind is SignalKind.NUMERIC
        assert is_event_rate(signal)


def test_signal_names_matches_registry_order() -> None:
    """signal_names returns the registry names in order."""
    assert signal_names() == tuple(signal.name for signal in REGISTRY)


@pytest.mark.parametrize("graph_id", [None, *GraphId])
def test_extractors_handle_each_graph_without_masking_errors(graph_id: GraphId | None) -> None:
    """Every registered extractor runs directly; missing graph data is intentional."""
    config = PactConfig()
    system = build_sil_system(config, ManualClock(), build_frames(1), plume_detector())
    state = system.apps.payload.initial_state()
    if graph_id is not None:
        key = ActivationKey(epoch=state.activation.expected_epoch, sequence=1)
        inputs = TickInputs(
            now_s=0.0,
            timestamp_utc="2026-06-01T00:00:00.000Z",
            activation_key=key,
            encoder=EncoderSample(sample_id="encoder", t_s=0.0, angle_rad=0.0),
            navigation=None,
            vision=None,
            health=HealthSample(feedback_valid=True, inhibit_confirmed=True, contained=False),
        )
        state = replace(
            state,
            activation=replace(
                state.activation,
                last=ActivationSnapshot(
                    key=key, graph_id=graph_id, previous_graph=None, reason="test"
                ),
            ),
            graph=runtime.initial_state(graph_id, inputs, GraphParameters(config=config)),
        )
    ctx = SampleContext(
        step=1,
        t=0.0,
        system=system,
        payload_state=state,
        fault_entries=system.apps.fault.initial_entries(),
        messages={},
        devices=sample_devices(system),
    )
    values = {signal.name: signal.extract(ctx) for signal in REGISTRY}
    assert values["system.mode"] == ("" if graph_id is None else graph_id.name)
    assert values["payload.graph"] == ("" if graph_id is None else graph_id.value)
    assert values["payload.node"] == ("" if state.graph is None else state.graph.node.value)
    if graph_id is not GraphId.OPERATE:
        for name in (
            "payload.e_hat",
            "payload.omega_t_res",
            "payload.residual_p00",
            "payload.residual_p11",
            "payload.residual_p_trace",
            "payload.aggregate_live",
            "payload.tracked_blobs",
        ):
            value = values[name]
            assert isinstance(value, float) and math.isnan(value), name
    assert "payload.gimbal_state" not in values
    assert "payload.current_target_id" not in values
    assert "fault.safety_mode" not in values
