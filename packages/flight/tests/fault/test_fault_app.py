"""Integration tests for the fault subsystem app (watchdog + fault routing over the bus)."""

from flight.fault.app import FaultApp
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import PactConfig
from flight.libs.messages import (
    FaultEventMsg,
    HeartbeatMsg,
    SafetyStateMsg,
    SystemModeActivatedMsg,
    SystemModeRequestMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import FaultCode, MessageType, SystemMode

_EPOCH = "epoch-fault-test"


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


def _activation(
    sequence: int,
    *,
    previous_mode: SystemMode | None = None,
    active_mode: SystemMode,
    request_id: str | None = None,
    recovery_authorized: bool = False,
    epoch: str = _EPOCH,
) -> SystemModeActivatedMsg:
    """Build an authority activation record on the app epoch."""
    return SystemModeActivatedMsg(
        msg_type=MessageType.SYSTEM_MODE_ACTIVATED,
        timestamp_utc="t",
        epoch=epoch,
        sequence=sequence,
        previous_mode=previous_mode,
        active_mode=active_mode,
        reason="test",
        request_id=request_id,
        recovery_authorized=recovery_authorized,
    )


def _app() -> tuple[FaultApp, MessageBus]:
    """Assemble a FaultApp monitoring 'payload' over a fresh bus and ManualClock."""
    bus = MessageBus()
    app = FaultApp.from_config(PactConfig(), bus, ManualClock(), ("payload",), _EPOCH)
    return app, bus


def test_heartbeats_keep_subsystem_alive() -> None:
    """A subsystem that keeps sending heartbeats never trips the watchdog."""
    app, bus = _app()
    requests = bus.subscribe(SystemModeRequestMsg)
    entries = app.initial_entries()
    now = 0.0
    for seq in range(5):
        now += 5.0
        bus.publish(_heartbeat("payload", seq))
        entries = app.tick(entries, now)
    assert entries["payload"].miss_count == 0
    assert requests.empty()


def test_silent_subsystem_requests_safe() -> None:
    """A subsystem that stops sending heartbeats trips the watchdog into a SAFE request."""
    app, bus = _app()
    requests = bus.subscribe(SystemModeRequestMsg)
    entries = app.initial_entries()
    now = 0.0
    for _ in range(3):  # watchdog_max_miss_count = 3
        now += 10.0  # > watchdog_interval_s (5.0) each tick, no heartbeats published
        entries = app.tick(entries, now)
    assert not requests.empty()
    request = requests.get_nowait()
    assert request.requested_mode is SystemMode.SAFE
    assert request.requested_by == "fault"


def test_fault_event_routed_to_safe_request() -> None:
    """A SAFE-triggering FaultEventMsg on the bus is routed to a SAFE request."""
    app, bus = _app()
    requests = bus.subscribe(SystemModeRequestMsg)
    entries = app.initial_entries()
    bus.publish(_fault(FaultCode.PROCESS_DIED))
    app.tick(entries, now=1.0)
    assert not requests.empty()
    assert requests.get_nowait().requested_mode is SystemMode.SAFE


def test_benign_fault_not_routed() -> None:
    """A non-SAFE fault (COMM_TIMEOUT) produces no system-mode request."""
    app, bus = _app()
    requests = bus.subscribe(SystemModeRequestMsg)
    entries = app.initial_entries()
    bus.publish(_fault(FaultCode.COMM_TIMEOUT))
    app.tick(entries, now=1.0)
    assert requests.empty()


def test_safety_evidence_carries_epoch_and_monotonic_sequence() -> None:
    """Each tick publishes SafetyStateMsg with the injected epoch + rising sequence."""
    app, bus = _app()
    safety = bus.subscribe(SafetyStateMsg)
    entries = app.initial_entries()
    app.tick(entries, now=1.0)
    app.tick(entries, now=2.0)
    first = safety.get_nowait()
    second = safety.get_nowait()
    assert first.evidence_epoch == _EPOCH
    assert second.evidence_sequence == first.evidence_sequence + 1
    assert second.observed_s == 2.0


def test_safe_latch_held_until_authorized_recovery() -> None:
    """SAFE latches on a triggering fault; an ordinary IDLE activation never releases it."""
    app, bus = _app()
    safety = bus.subscribe(SafetyStateMsg)
    entries = app.initial_entries()
    bus.publish(_fault(FaultCode.PROCESS_DIED))
    entries = app.tick(entries, now=1.0)
    latched = _drain_last(safety)
    assert latched is not None and latched.safe_latched
    bus.publish(_activation(1, active_mode=SystemMode.IDLE))
    entries = app.tick(entries, now=2.0)
    held = _drain_last(safety)
    assert held is not None and held.safe_latched


def test_authorized_recovery_releases_latch_and_stamps_request() -> None:
    """An authorized SAFE->IDLE recovery releases the latch and carries request_id."""
    app, bus = _app()
    safety = bus.subscribe(SafetyStateMsg)
    entries = app.initial_entries()
    bus.publish(_fault(FaultCode.PROCESS_DIED))
    entries = app.tick(entries, now=1.0)
    bus.publish(
        _activation(
            1,
            previous_mode=SystemMode.SAFE,
            active_mode=SystemMode.IDLE,
            request_id="rec-1",
            recovery_authorized=True,
        )
    )
    entries = app.tick(entries, now=2.0)
    released = _drain_last(safety)
    assert released is not None
    assert not released.safe_latched
    assert released.recovery_request_id == "rec-1"


def _drain_last(safety: Subscription[SafetyStateMsg]) -> SafetyStateMsg | None:
    """Return the newest SafetyStateMsg on the subscription, or None."""
    latest: SafetyStateMsg | None = None
    while not safety.empty():
        latest = safety.get_nowait()
    return latest
