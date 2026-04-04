"""Heartbeat watchdog — detects silent process death via missed heartbeats.

Maintains a dict of WatchdogEntry records, one per monitored subsystem.  On each
check_heartbeats() call, entries whose last_heartbeat_time is older than max_interval_s
have their miss_count incremented.  When miss_count exceeds the configured threshold
(fault.watchdog_max_miss_count), a FaultEventMsg(WATCHDOG_EXPIRE) is emitted.

Satisfies: REQ-SAFE-HIGH-002.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass
from datetime import datetime, timezone

# internal
from pact.types.enums import FaultCode, MessageType
from pact.types.messages import FaultEventMsg


@dataclass(frozen=True)
class WatchdogEntry:
    """State record for one monitored subsystem.

    subsystem:           Name of the subsystem being watched (matches HeartbeatMsg.subsystem).
    last_heartbeat_time: Unix timestamp (float, seconds) of the most recent heartbeat.
    max_interval_s:      Maximum allowed seconds between heartbeats before miss is counted.
    miss_count:          Consecutive missed heartbeat intervals since the last received beat.
    """

    subsystem: str
    last_heartbeat_time: float
    max_interval_s: float
    miss_count: int


def check_heartbeats(
    entries: dict[str, WatchdogEntry],
    now: float,
    max_miss_count: int,
) -> tuple[dict[str, WatchdogEntry], list[FaultEventMsg]]:
    """Increment miss counts for overdue subsystems and emit faults at threshold.

    For each WatchdogEntry:
      - If (now - last_heartbeat_time) > max_interval_s, increment miss_count.
      - If miss_count >= max_miss_count, emit a FaultEventMsg(WATCHDOG_EXPIRE).
      - Otherwise, carry the entry forward unchanged.

    Returns:
        updated_entries: dict with miss_count incremented for overdue entries.
        faults:          list of FaultEventMsg for entries that exceeded the threshold.

    Note: entries that emit a fault are NOT removed from the dict — the fault process
    decides how to handle them (e.g. enter safe mode).
    """
    now_str = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    updated: dict[str, WatchdogEntry] = {}
    faults: list[FaultEventMsg] = []

    for name, entry in entries.items():
        elapsed = now - entry.last_heartbeat_time
        if elapsed > entry.max_interval_s:
            new_miss = entry.miss_count + 1
            updated[name] = WatchdogEntry(
                subsystem=entry.subsystem,
                last_heartbeat_time=entry.last_heartbeat_time,
                max_interval_s=entry.max_interval_s,
                miss_count=new_miss,
            )
            if new_miss >= max_miss_count:
                faults.append(
                    FaultEventMsg(
                        msg_type=MessageType.FAULT_EVENT,
                        timestamp_utc=now_str,
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
