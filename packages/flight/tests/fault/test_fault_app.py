"""Integration tests for the fault subsystem app (watchdog + fault routing over the bus)."""

from flight.fault.app import FaultApp
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import (
    ActivationKey,
    FaultEventMsg,
    HeartbeatMsg,
    ModeChangeMsg,
    SafetyStateMsg,
    SystemModeActivatedMsg,
    SystemModeRequestMsg,
)
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


def _activation(
    mode: SystemMode, sequence: int, recovery: bool = False, epoch: str = "e"
) -> SystemModeActivatedMsg:
    """Build an authority activation."""
    return SystemModeActivatedMsg(
        msg_type=MessageType.SYSTEM_MODE_ACTIVATED,
        timestamp_utc="t",
        key=ActivationKey(epoch, sequence),
        previous_mode=None,
        active_mode=mode,
        reason="test",
        request_id=f"r{sequence}",
        recovery_authorized=recovery,
    )


def _latest_safety(sub: object) -> SafetyStateMsg:
    """Drain a SafetyStateMsg subscription and return the newest message."""
    last = None
    while not sub.empty():  # type: ignore[attr-defined]
        last = sub.get_nowait()  # type: ignore[attr-defined]
    assert isinstance(last, SafetyStateMsg)
    return last


def test_safe_fault_requests_safe_from_authority() -> None:
    """A SAFE-triggering fault latches immediately and sends one SAFE request."""
    app, bus = _app()
    requests = bus.subscribe(SystemModeRequestMsg)
    safety = bus.subscribe(SafetyStateMsg)
    entries = app.initial_entries()
    bus.publish(_fault(FaultCode.PROCESS_DIED))
    bus.publish(_fault(FaultCode.CAMERA_STALL))
    app.tick(entries, now=1.0)
    request = requests.get_nowait()
    assert request.requested_mode is SystemMode.SAFE
    assert request.requested_by == "fault"
    assert requests.empty()  # one request per tick
    assert _latest_safety(safety).safe_latched


def test_no_safe_request_once_authority_is_safe() -> None:
    """While the authority reports SAFE, further SAFE faults latch but do not re-request."""
    app, bus = _app()
    requests = bus.subscribe(SystemModeRequestMsg)
    entries = app.initial_entries()
    bus.publish(_activation(SystemMode.SAFE, 1))
    bus.publish(_fault(FaultCode.PROCESS_DIED))
    app.tick(entries, now=1.0)
    assert requests.empty()
    assert app.safety.safe_latched


def test_recovery_activation_releases_latch_when_clear() -> None:
    """A recovery-authorized activation clears the latch and publishes ModeChangeMsg(IDLE)."""
    app, bus = _app()
    modes = bus.subscribe(ModeChangeMsg)
    entries = app.initial_entries()
    bus.publish(_fault(FaultCode.PROCESS_DIED))
    entries = app.tick(entries, now=1.0)
    bus.publish(_activation(SystemMode.SAFE, 1))
    bus.publish(_heartbeat("payload", 0))
    entries = app.tick(entries, now=2.0)
    while not modes.empty():
        modes.get_nowait()
    bus.publish(_activation(SystemMode.IDLE, 2, recovery=True))
    bus.publish(_heartbeat("payload", 1))
    app.tick(entries, now=3.0)
    assert not app.safety.safe_latched
    assert app.safety.safe_reason is FaultCode.NONE
    assert modes.get_nowait().new_mode is SystemMode.IDLE


def test_recovery_activation_refused_if_fault_fires_same_tick() -> None:
    """A recovery activation does not clear the latch when a SAFE fault fires that tick."""
    app, bus = _app()
    requests = bus.subscribe(SystemModeRequestMsg)
    entries = app.initial_entries()
    bus.publish(_fault(FaultCode.PROCESS_DIED))
    entries = app.tick(entries, now=1.0)
    while not requests.empty():
        requests.get_nowait()
    bus.publish(_activation(SystemMode.IDLE, 2, recovery=True))
    bus.publish(_fault(FaultCode.PROCESS_DIED))
    app.tick(entries, now=2.0)
    assert app.safety.safe_latched
    assert requests.get_nowait().requested_mode is SystemMode.SAFE  # re-request SAFE


def test_non_recovery_activation_does_not_release_latch() -> None:
    """Only a recovery-authorized activation may clear the latch."""
    app, bus = _app()
    entries = app.initial_entries()
    bus.publish(_fault(FaultCode.PROCESS_DIED))
    entries = app.tick(entries, now=1.0)
    bus.publish(_activation(SystemMode.IDLE, 2))
    bus.publish(_heartbeat("payload", 0))
    app.tick(entries, now=2.0)
    assert app.safety.safe_latched


def test_stale_recovery_replay_is_ignored() -> None:
    """A replayed recovery activation older than the last applied key cannot clear the latch."""
    app, bus = _app()
    entries = app.initial_entries()
    bus.publish(_activation(SystemMode.IDLE, 2, recovery=True))
    bus.publish(_activation(SystemMode.SAFE, 3))
    bus.publish(_fault(FaultCode.PROCESS_DIED))
    entries = app.tick(entries, now=1.0)
    bus.publish(_activation(SystemMode.IDLE, 2, recovery=True))  # snapshot replay of old key
    bus.publish(_heartbeat("payload", 0))
    app.tick(entries, now=2.0)
    assert app.safety.safe_latched
