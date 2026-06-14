"""Tests for the thermal housekeeping app (telemetry + threshold fault + command ack)."""

from flight.hal.drivers_sim import SimScalarSensor
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    FaultEventMsg,
    RoutedCommandMsg,
    TelemetryEventMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import AckStatus, FaultCode, MessageType
from flight.thermal.app import ThermalApp


def _routed(
    command_id: str, target: str, params: dict[str, object] | None = None
) -> RoutedCommandMsg:
    """Build a RoutedCommandMsg envelope for command-handling tests."""
    return RoutedCommandMsg(
        msg_type=MessageType.ROUTED_COMMAND,
        timestamp_utc="t",
        target=target,
        command_id=command_id,
        params=params or {},  # type: ignore[arg-type]
        source="ground",
        seq=1,
    )


def _app(readings: list[float]) -> tuple[ThermalApp, MessageBus]:
    """Assemble a ThermalApp over a scripted temperature sensor and a fresh bus."""
    bus = MessageBus()
    sensor = SimScalarSensor(readings)
    app = ThermalApp.from_config(PactConfig(), bus, ManualClock(), sensor)
    return app, bus


def test_nominal_reading_publishes_telemetry_no_fault() -> None:
    """A temperature below the limit publishes telemetry and no fault."""
    app, bus = _app([25.0])
    telem = bus.subscribe(TelemetryEventMsg)
    fault = bus.subscribe(FaultEventMsg)
    app.sample()
    assert not telem.empty()
    event = telem.get_nowait()
    assert event.subsystem == "thermal"
    assert event.payload["temperature_c"] == 25.0
    assert fault.empty()


def test_over_limit_reading_publishes_thermal_fault() -> None:
    """A temperature above thermal_limit_c (80.0) publishes THERMAL_OVER_LIMIT."""
    app, bus = _app([95.0])
    fault = bus.subscribe(FaultEventMsg)
    app.sample()
    assert not fault.empty()
    event = fault.get_nowait()
    assert event.fault_code is FaultCode.THERMAL_OVER_LIMIT
    assert event.subsystem == "thermal"


def test_set_thermal_limit_executes_and_acks() -> None:
    """A routed SET_THERMAL_LIMIT applies the new limit and produces an ACCEPTED exec ack."""
    app, bus = _app([25.0])
    acks = bus.subscribe(CommandAckMsg)
    bus.publish(_routed("SET_THERMAL_LIMIT", "thermal", {"limit_c": 20.0}))
    app.handle_commands()
    ack = acks.get_nowait()
    assert ack.status is AckStatus.ACCEPTED
    assert ack.command_id == "SET_THERMAL_LIMIT"
    # The applied limit (20C) now drives an over-limit fault at 25C.
    fault = bus.subscribe(FaultEventMsg)
    app.sample()
    assert not fault.empty()
    assert fault.get_nowait().fault_code is FaultCode.THERMAL_OVER_LIMIT


def test_unsupported_command_for_thermal_is_rejected() -> None:
    """A routed command thermal does not implement is acked REJECTED (no silent drop)."""
    app, bus = _app([25.0])
    acks = bus.subscribe(CommandAckMsg)
    bus.publish(_routed("PING", "thermal"))
    app.handle_commands()
    ack = acks.get_nowait()
    assert ack.status is AckStatus.REJECTED


def test_command_for_other_subsystem_ignored() -> None:
    """A routed command targeting another subsystem produces no ack."""
    app, bus = _app([25.0])
    acks = bus.subscribe(CommandAckMsg)
    bus.publish(_routed("PING", "payload"))
    app.handle_commands()
    assert acks.empty()
