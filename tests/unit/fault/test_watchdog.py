"""Unit tests for pact.fault.watchdog — check_heartbeats() and WatchdogEntry.

Satisfies: §6.2 of PACT_SW_ARCH.md — Fault Detection subsystem unit tests.
REQ-SAFE-HIGH-002, GOAL-006
"""

from __future__ import annotations

# third-party
import pytest

# module under test
from pact.fault.watchdog import WatchdogEntry, check_heartbeats

# pact types
from pact.types.enums import FaultCode
from pact.types.messages import FaultEventMsg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_entry(
    subsystem: str = "inference",
    last_heartbeat_time: float = 1000.0,
    max_interval_s: float = 5.0,
    miss_count: int = 0,
) -> WatchdogEntry:
    """Construct a WatchdogEntry for watchdog tests."""
    return WatchdogEntry(
        subsystem=subsystem,
        last_heartbeat_time=last_heartbeat_time,
        max_interval_s=max_interval_s,
        miss_count=miss_count,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fresh_heartbeat_no_fault() -> None:
    """A subsystem with a recent heartbeat must not produce any faults."""
    now = 1004.0  # 4s after last heartbeat; interval = 5s — not yet overdue
    entries = {"inference": make_entry(last_heartbeat_time=1000.0, max_interval_s=5.0)}

    updated_entries, faults = check_heartbeats(entries, now=now, max_miss_count=3)
    assert faults == [], f"Expected no faults for fresh heartbeat, got {faults}"
    # miss_count must not increase for a fresh heartbeat
    assert updated_entries["inference"].miss_count == 0, (
        f"miss_count should remain 0 for fresh heartbeat, "
        f"got {updated_entries['inference'].miss_count}"
    )


def test_overdue_heartbeat_increments_miss_count() -> None:
    """A subsystem overdue by more than max_interval_s must have its miss_count incremented."""
    now = 1010.0  # 10s after last heartbeat; interval = 5s — definitely overdue
    entries = {
        "inference": make_entry(
            last_heartbeat_time=1000.0,
            max_interval_s=5.0,
            miss_count=0,
        )
    }

    updated_entries, faults = check_heartbeats(entries, now=now, max_miss_count=3)
    assert updated_entries["inference"].miss_count == 1, (
        f"Expected miss_count=1 after one overdue check, "
        f"got {updated_entries['inference'].miss_count}"
    )


def test_exceeded_miss_count_emits_fault() -> None:
    """A subsystem that has exceeded max_miss_count must emit a FaultEventMsg.

    Default max_miss_count = 3 (from FaultConfig). When miss_count reaches 3,
    a WATCHDOG_EXPIRE fault must be emitted.
    """
    now = 1010.0
    # Already at miss_count=2; this call should increment to 3 and emit fault
    entries = {
        "inference": make_entry(
            last_heartbeat_time=1000.0,
            max_interval_s=5.0,
            miss_count=2,  # one below the threshold of 3
        )
    }

    updated_entries, faults = check_heartbeats(entries, now=now, max_miss_count=3)

    fault_codes = [f.fault_code for f in faults]
    assert FaultCode.WATCHDOG_EXPIRE in fault_codes, (
        f"Expected WATCHDOG_EXPIRE fault when miss_count exceeds threshold, "
        f"got {fault_codes}"
    )


def test_fresh_subsystem_no_miss_increment() -> None:
    """Multiple subsystems: only the overdue one should have its miss_count incremented."""
    now = 1010.0
    entries = {
        "inference": make_entry(  # overdue
            subsystem="inference", last_heartbeat_time=1000.0, max_interval_s=5.0
        ),
        "storage": make_entry(  # fresh
            subsystem="storage", last_heartbeat_time=1009.0, max_interval_s=5.0
        ),
    }

    updated_entries, _ = check_heartbeats(entries, now=now, max_miss_count=3)
    assert updated_entries["inference"].miss_count == 1, "inference should be incremented"
    assert updated_entries["storage"].miss_count == 0, "storage should not be incremented"


def test_faults_are_fault_event_msgs() -> None:
    """All faults emitted by check_heartbeats must be FaultEventMsg instances."""
    now = 1010.0
    entries = {
        "inference": make_entry(last_heartbeat_time=1000.0, max_interval_s=5.0, miss_count=2)
    }
    _, faults = check_heartbeats(entries, now=now, max_miss_count=3)
    for fault in faults:
        assert isinstance(fault, FaultEventMsg), (
            f"Expected FaultEventMsg, got {type(fault)}: {fault}"
        )


def test_check_heartbeats_empty_entries() -> None:
    """check_heartbeats with no entries must return empty dict and no faults."""
    updated, faults = check_heartbeats({}, now=1000.0, max_miss_count=3)
    assert updated == {}
    assert faults == []


def test_watchdog_entry_immutable() -> None:
    """WatchdogEntry is a frozen dataclass — direct mutation must raise AttributeError."""
    entry = make_entry()
    with pytest.raises(AttributeError):
        entry.miss_count = 99  # type: ignore[misc]
