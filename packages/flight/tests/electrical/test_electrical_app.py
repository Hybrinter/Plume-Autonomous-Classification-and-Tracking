"""Tests for the electrical housekeeping app (telemetry + threshold fault + command ack)."""

from flight.electrical.app import ElectricalApp
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


def _app(readings: list[float]) -> tuple[ElectricalApp, MessageBus]:
    """Assemble an ElectricalApp over a scripted power sensor and a fresh bus."""
    bus = MessageBus()
    sensor = SimScalarSensor(readings)
    app = ElectricalApp.from_config(PactConfig(), bus, ManualClock(), sensor)
    return app, bus


def test_nominal_reading_publishes_telemetry_no_fault() -> None:
    """A power draw below the limit publishes telemetry and no fault."""
    app, bus = _app([30.0])
    telem = bus.subscribe(TelemetryEventMsg)
    fault = bus.subscribe(FaultEventMsg)
    app.sample()
    assert not telem.empty()
    event = telem.get_nowait()
    assert event.subsystem == "electrical"
    assert event.payload["power_w"] == 30.0
    assert fault.empty()


def test_over_limit_reading_publishes_power_fault() -> None:
    """A power draw above power_limit_w (55.0) publishes POWER_OVER_LIMIT."""
    app, bus = _app([70.0])
    fault = bus.subscribe(FaultEventMsg)
    app.sample()
    assert not fault.empty()
    event = fault.get_nowait()
    assert event.fault_code is FaultCode.POWER_OVER_LIMIT
    assert event.subsystem == "electrical"


def test_command_targeting_electrical_is_rejected() -> None:
    """A routed command targeting 'electrical' is acked REJECTED (no commandable behavior)."""
    app, bus = _app([30.0])
    acks = bus.subscribe(CommandAckMsg)
    bus.publish(
        RoutedCommandMsg(
            msg_type=MessageType.ROUTED_COMMAND,
            timestamp_utc="t",
            target="electrical",
            command_id="PING",
            params={},
            source="ground",
            seq=1,
        )
    )
    app.handle_commands()
    ack = acks.get_nowait()
    assert ack.status is AckStatus.REJECTED
    assert ack.command_id == "PING"
