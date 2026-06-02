"""Fault (FDIR) subsystem: heartbeat watchdog, fault-to-mode policy, and the FDIR app."""

from flight.fault.app import FaultApp
from flight.fault.policy import (
    SAFE_TRIGGERING_FAULTS,
    decide_mode_change,
    enter_safe_mode,
    exit_safe_mode,
)
from flight.fault.watchdog import WatchdogEntry, build_entries, check_heartbeats

__all__ = [
    "SAFE_TRIGGERING_FAULTS",
    "FaultApp",
    "WatchdogEntry",
    "build_entries",
    "check_heartbeats",
    "decide_mode_change",
    "enter_safe_mode",
    "exit_safe_mode",
]
