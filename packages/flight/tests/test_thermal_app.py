"""Tests for the thermal housekeeping app (telemetry + threshold fault + command ack)."""

from flight.hal.drivers_sim import SimScalarSensor
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import CommandMsg, FaultEventMsg, TelemetryEventMsg
from flight.libs.time import ManualClock
from flight.libs.types import FaultCode, MessageType
from flight.thermal.app import ThermalApp


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


def test_command_targeting_thermal_is_acknowledged() -> None:
    """A CommandMsg targeting 'thermal' produces a command_ack telemetry event."""
    app, bus = _app([25.0])
    telem = bus.subscribe(TelemetryEventMsg)
    bus.publish(
        CommandMsg(
            msg_type=MessageType.COMMAND,
            timestamp_utc="t",
            target="thermal",
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


def test_command_for_other_subsystem_ignored() -> None:
    """A CommandMsg targeting another subsystem produces no ack."""
    app, bus = _app([25.0])
    telem = bus.subscribe(TelemetryEventMsg)
    bus.publish(
        CommandMsg(
            msg_type=MessageType.COMMAND,
            timestamp_utc="t",
            target="payload",
            command_id="ping",
            params={},
            source="ground",
            seq=1,
        )
    )
    app.handle_commands()
    assert telem.empty()
