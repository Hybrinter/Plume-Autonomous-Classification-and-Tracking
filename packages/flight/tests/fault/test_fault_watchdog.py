"""Tests for the pure heartbeat watchdog."""

from flight.fault.watchdog import build_entries, check_heartbeats
from flight.libs.messages import FaultEventMsg
from flight.libs.types import FaultCode


def test_fresh_entries_have_no_misses() -> None:
    """build_entries starts every subsystem at zero misses."""
    entries = build_entries(("payload",), max_interval_s=5.0, now=100.0)
    assert entries["payload"].miss_count == 0


def test_recent_heartbeat_not_overdue() -> None:
    """A subsystem within max_interval_s is not counted as a miss."""
    entries = build_entries(("payload",), max_interval_s=5.0, now=0.0)
    updated, faults = check_heartbeats(entries, now=3.0, max_miss_count=3, now_iso="t")
    assert updated["payload"].miss_count == 0
    assert faults == []


def test_overdue_increments_miss_without_fault_below_threshold() -> None:
    """One overdue interval increments the miss count but raises no fault yet."""
    entries = build_entries(("payload",), max_interval_s=5.0, now=0.0)
    updated, faults = check_heartbeats(entries, now=6.0, max_miss_count=3, now_iso="t")
    assert updated["payload"].miss_count == 1
    assert faults == []


def test_threshold_emits_watchdog_expire() -> None:
    """Reaching max_miss_count consecutive overdue intervals emits WATCHDOG_EXPIRE."""
    entries = build_entries(("payload",), max_interval_s=5.0, now=0.0)
    now = 0.0
    faults: list[FaultEventMsg] = []
    for _ in range(3):  # max_miss_count = 3
        now += 6.0
        entries, faults = check_heartbeats(entries, now, max_miss_count=3, now_iso="t")
    assert entries["payload"].miss_count == 3
    assert len(faults) == 1
    assert faults[0].fault_code is FaultCode.WATCHDOG_EXPIRE
    assert faults[0].subsystem == "payload"
