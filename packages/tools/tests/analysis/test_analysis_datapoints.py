"""Tests for the analysis signal registry (datapoints)."""

from tools.analysis.datapoints import (
    GROUPS,
    MESSAGE_TYPES,
    REGISTRY,
    Signal,
    SignalKind,
    accumulable_names,
    is_event_rate,
    signal_names,
    signals_for_group,
)

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
    "mechanical",
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
