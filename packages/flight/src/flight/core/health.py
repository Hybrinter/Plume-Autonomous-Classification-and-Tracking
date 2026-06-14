"""Startup health gate: require every monitored subsystem to heartbeat before NOMINAL.

At startup the composition root waits a bounded window for a first heartbeat from each monitored
subsystem. If any are still silent when the window closes, the system enters SAFE and annunciates
rather than declaring NOMINAL on a half-initialized topology (spec Section 7). The decision is a
pure set comparison so it is fully unit-tested; flight.core.main owns the time-bounded collection.

Contains:
  - missing_heartbeats: the monitored subsystems that have not yet been seen.
  - startup_healthy: True iff every monitored subsystem has been seen.

Satisfies: REQ-OPER-HIGH-002, REQ-SAFE-HIGH-002, REQ-PLAT-SUP-001.
"""

from __future__ import annotations


def missing_heartbeats(seen: set[str], monitored: tuple[str, ...]) -> set[str]:
    """Return the monitored subsystems that have not produced a heartbeat yet (pure).

    Args:
        seen: Subsystem names observed to have heartbeat at least once.
        monitored: The subsystems that must heartbeat for a healthy startup.

    Returns:
        The set of monitored names absent from seen (empty when startup is healthy).
    """
    return set(monitored) - seen


def startup_healthy(seen: set[str], monitored: tuple[str, ...]) -> bool:
    """Return True iff every monitored subsystem has heartbeat at least once (pure).

    Args:
        seen: Subsystem names observed to have heartbeat at least once.
        monitored: The subsystems that must heartbeat for a healthy startup.

    Returns:
        True when no monitored subsystem is missing, else False (the caller enters SAFE).
    """
    return not missing_heartbeats(seen, monitored)
