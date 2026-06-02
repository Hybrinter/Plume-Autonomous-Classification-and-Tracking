"""Heartbeat watchdog: detects silent subsystem death via missed heartbeats.

Pure functions over a dict of WatchdogEntry (one per monitored subsystem). On each
check_heartbeats() call, entries whose last_heartbeat_time is older than max_interval_s
have miss_count incremented; at max_miss_count a FaultEventMsg(WATCHDOG_EXPIRE) is
emitted. The caller owns the clock and the entries dict (state threaded in and out);
this module performs no I/O and reads no clock -- timestamps are injected.

Contains:
  - WatchdogEntry: frozen per-subsystem record (subsystem, last_heartbeat_time in
    monotonic seconds, max_interval_s, miss_count).
  - build_entries: construct the starting entries dict from a tuple of subsystem names.
  - check_heartbeats: increment misses for overdue subsystems and emit WATCHDOG_EXPIRE
    faults at the configured threshold; returns the updated dict and the faults list.

Satisfies: REQ-SAFE-HIGH-002.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass, replace

# internal
from flight.libs.messages import FaultEventMsg
from flight.libs.types import FaultCode, MessageType


@dataclass(frozen=True, slots=True)
class WatchdogEntry:
    """Immutable watchdog record for one monitored subsystem.

    Attributes:
        subsystem: Name matching HeartbeatMsg.subsystem.
        last_heartbeat_time: Monotonic seconds of the most recent received heartbeat.
        max_interval_s: Maximum allowed seconds between heartbeats before a miss counts.
        miss_count: Consecutive overdue intervals since the last received heartbeat.
    """

    subsystem: str
    last_heartbeat_time: float
    max_interval_s: float
    miss_count: int


def build_entries(
    subsystems: tuple[str, ...],
    max_interval_s: float,
    now: float,
) -> dict[str, WatchdogEntry]:
    """Construct the starting watchdog entries dict.

    Each subsystem starts with last_heartbeat_time=now and miss_count=0, giving every
    subsystem a full interval to send its first heartbeat.

    Args:
        subsystems: Names of the subsystems to monitor.
        max_interval_s: Maximum seconds between heartbeats before a miss is counted.
        now: Current monotonic seconds (used as the initial last_heartbeat_time).

    Returns:
        A dict mapping each subsystem name to a fresh WatchdogEntry.
    """
    return {
        name: WatchdogEntry(
            subsystem=name,
            last_heartbeat_time=now,
            max_interval_s=max_interval_s,
            miss_count=0,
        )
        for name in subsystems
    }


def check_heartbeats(
    entries: dict[str, WatchdogEntry],
    now: float,
    max_miss_count: int,
    now_iso: str,
) -> tuple[dict[str, WatchdogEntry], list[FaultEventMsg]]:
    """Increment miss counts for overdue subsystems and emit faults at the threshold.

    For each entry: if (now - last_heartbeat_time) > max_interval_s, increment
    miss_count; if the new miss_count >= max_miss_count, emit a
    FaultEventMsg(WATCHDOG_EXPIRE). Entries that emitted a fault are NOT removed -- the
    caller decides how to respond (e.g. request SAFE mode).

    Args:
        entries: Current watchdog entries (threaded state; not mutated in place).
        now: Current monotonic seconds.
        max_miss_count: Consecutive overdue intervals required to emit WATCHDOG_EXPIRE.
        now_iso: Wall-clock ISO timestamp to stamp on any emitted FaultEventMsg.

    Returns:
        (updated_entries, faults): the new entries dict and any WATCHDOG_EXPIRE faults.
    """
    updated: dict[str, WatchdogEntry] = {}
    faults: list[FaultEventMsg] = []

    for name, entry in entries.items():
        elapsed = now - entry.last_heartbeat_time
        if elapsed > entry.max_interval_s:
            new_miss = entry.miss_count + 1
            updated[name] = replace(entry, miss_count=new_miss)
            if new_miss >= max_miss_count:
                faults.append(
                    FaultEventMsg(
                        msg_type=MessageType.FAULT_EVENT,
                        timestamp_utc=now_iso,
                        fault_code=FaultCode.WATCHDOG_EXPIRE,
                        subsystem=name,
                        detail=(
                            f"watchdog expired: {new_miss} consecutive misses "
                            f"(max_interval_s={entry.max_interval_s})"
                        ),
                    )
                )
        else:
            updated[name] = entry

    return updated, faults
