"""Tests for the electrical housekeeping app (telemetry + threshold fault + command ack)."""

from flight.electrical.app import ElectricalApp
from flight.hal.drivers_sim import SimScalarSensor
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import CommandMsg, FaultEventMsg, TelemetryEventMsg
from flight.libs.time import ManualClock
from flight.libs.types import FaultCode, MessageType


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


def test_command_targeting_electrical_is_acknowledged() -> None:
    """A CommandMsg targeting 'electrical' produces a command_ack telemetry event."""
    app, bus = _app([30.0])
    telem = bus.subscribe(TelemetryEventMsg)
    bus.publish(
        CommandMsg(
            msg_type=MessageType.COMMAND,
            timestamp_utc="t",
            target="electrical",
            command_id="ping",
            params={},
            source="ground",
            seq=1,
        )
    )
    app.handle_commands()
    assert not telem.empty()
    ack = telem.get_nowait()
    assert ack.event_name == "command_ack"
    assert ack.payload["command_id"] == "ping"
