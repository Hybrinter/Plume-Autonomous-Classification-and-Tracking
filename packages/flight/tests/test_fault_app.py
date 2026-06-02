"""Integration tests for the fault subsystem app (watchdog + fault routing over the bus)."""

from flight.fault.app import FaultApp
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import FaultEventMsg, HeartbeatMsg, ModeChangeMsg
from flight.libs.time import ManualClock
from flight.libs.types import FaultCode, MessageType, SystemMode


def _heartbeat(subsystem: str, seq: int) -> HeartbeatMsg:
    """Build a HeartbeatMsg for the given subsystem and sequence number."""
    return HeartbeatMsg(
        msg_type=MessageType.HEARTBEAT,
        timestamp_utc="t",
        subsystem=subsystem,
        sequence=seq,
    )


def _fault(code: FaultCode) -> FaultEventMsg:
    """Build a FaultEventMsg carrying the given fault code from the payload subsystem."""
    return FaultEventMsg(
        msg_type=MessageType.FAULT_EVENT,
        timestamp_utc="t",
        fault_code=code,
        subsystem="payload",
        detail="",
    )


def _app() -> tuple[FaultApp, MessageBus]:
    """Assemble a FaultApp monitoring 'payload' over a fresh bus and ManualClock."""
    bus = MessageBus()
    app = FaultApp.from_config(PactConfig(), bus, ManualClock(), ("payload",))
    return app, bus


def test_heartbeats_keep_subsystem_alive() -> None:
    """A subsystem that keeps sending heartbeats never trips the watchdog."""
    app, bus = _app()
    mode_sub = bus.subscribe(ModeChangeMsg)
    entries = app.initial_entries()
    now = 0.0
    for seq in range(5):
        now += 5.0
        bus.publish(_heartbeat("payload", seq))
        entries = app.tick(entries, now)
    assert entries["payload"].miss_count == 0
    assert mode_sub.empty()


def test_silent_subsystem_triggers_safe() -> None:
    """A subsystem that stops sending heartbeats trips the watchdog into SAFE."""
    app, bus = _app()
    mode_sub = bus.subscribe(ModeChangeMsg)
    entries = app.initial_entries()
    now = 0.0
    for _ in range(3):  # watchdog_max_miss_count = 3
        now += 10.0  # > watchdog_interval_s (5.0) each tick, no heartbeats published
        entries = app.tick(entries, now)
    assert not mode_sub.empty()
    assert mode_sub.get_nowait().new_mode is SystemMode.SAFE


def test_fault_event_routed_to_safe() -> None:
    """A SAFE-triggering FaultEventMsg on the bus is routed to a ModeChangeMsg(SAFE)."""
    app, bus = _app()
    mode_sub = bus.subscribe(ModeChangeMsg)
    entries = app.initial_entries()
    bus.publish(_fault(FaultCode.PROCESS_DIED))
    app.tick(entries, now=1.0)
    assert not mode_sub.empty()
    assert mode_sub.get_nowait().new_mode is SystemMode.SAFE


def test_benign_fault_not_routed() -> None:
    """A non-SAFE fault (COMM_TIMEOUT) produces no mode change."""
    app, bus = _app()
    mode_sub = bus.subscribe(ModeChangeMsg)
    entries = app.initial_entries()
    bus.publish(_fault(FaultCode.COMM_TIMEOUT))
    app.tick(entries, now=1.0)
    assert mode_sub.empty()
