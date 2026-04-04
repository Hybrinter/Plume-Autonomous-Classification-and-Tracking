"""Fault process entry point.

Runs as a multiprocessing.Process.  Performs three duties on a configurable timer:
  1. Watchdog: calls check_heartbeats() to detect silent process death.
  2. Fault dispatch: drains fault_queue and calls FAULT_HANDLERS[fault_code](event).
  3. Mode propagation: puts any returned ModeChangeMsg on mode_queue.

Startup invariant (asserted immediately on entry):
  FAULT_HANDLERS must contain a handler for every FaultCode member.  This is a
  programming error and will crash the process immediately, before any subsystem
  is allowed to produce faults.

Satisfies: REQ-SAFE-HIGH-002, REQ-GIMB-HIGH-003, GOAL-006.
"""

from __future__ import annotations

# stdlib
import dataclasses
import multiprocessing
import time
from typing import Optional

# internal
from pact.types.config import FaultConfig
from pact.types.enums import FaultCode, MessageType
from pact.types.messages import FaultEventMsg, HeartbeatMsg, ModeChangeMsg
from pact.fault.handlers import FAULT_HANDLERS
from pact.fault.watchdog import WatchdogEntry, check_heartbeats

import structlog

log = structlog.get_logger().bind(subsystem="fault")

# Subsystem names that the watchdog monitors.  The fault process creates one WatchdogEntry
# per name.  This list must match the subsystem strings in HeartbeatMsg.subsystem.
MONITORED_SUBSYSTEMS: tuple[str, ...] = (
    "imaging",
    "inference",
    "controller",
    "storage",
    "telemetry",
    "comms",
)


def run_fault_process(
    config: FaultConfig,
    heartbeat_queue: "multiprocessing.Queue[HeartbeatMsg]",
    fault_queue: "multiprocessing.Queue[FaultEventMsg]",
    mode_queue: "multiprocessing.Queue[ModeChangeMsg]",
) -> None:
    """Fault process main loop.

    Startup:
      - Asserts FAULT_HANDLERS covers all FaultCode members (programming-error guard).
      - Initialises one WatchdogEntry per monitored subsystem.

    Main loop (runs every config.watchdog_interval_s seconds approximately):
      1. Drain heartbeat_queue and update WatchdogEntry.last_heartbeat_time.
      2. Drain fault_queue and dispatch each event to its handler.
      3. Run check_heartbeats() and dispatch any WATCHDOG_EXPIRE faults.
      4. Put any ModeChangeMsg results on mode_queue.

    # TODO: implement graceful shutdown via a stop_event or sentinel value.
    # TODO: add per-subsystem fault count tracking and escalation policy.
    """
    # --- startup invariant: handler completeness check ---
    missing: list[FaultCode] = [fc for fc in FaultCode if fc not in FAULT_HANDLERS]
    assert not missing, (
        f"FAULT_HANDLERS is incomplete — missing handlers for: {[fc.value for fc in missing]}"
    )
    log.info("fault_handlers_verified", count=len(FAULT_HANDLERS))

    # --- initialise watchdog entries ---
    now = time.monotonic()
    watchdog_entries: dict[str, WatchdogEntry] = {
        name: WatchdogEntry(
            subsystem=name,
            last_heartbeat_time=now,        # give subsystems full interval to send first beat
            max_interval_s=config.watchdog_interval_s,
            miss_count=0,
        )
        for name in MONITORED_SUBSYSTEMS
    }

    log.info("fault_process_started", monitored=MONITORED_SUBSYSTEMS)

    while True:
        loop_start = time.monotonic()

        # --- 1. drain heartbeats ---
        while True:
            try:
                hb: HeartbeatMsg = heartbeat_queue.get_nowait()
                if hb.subsystem in watchdog_entries:
                    entry = watchdog_entries[hb.subsystem]
                    watchdog_entries[hb.subsystem] = dataclasses.replace(
                        entry,
                        last_heartbeat_time=time.monotonic(),
                        miss_count=0,
                    )
                    log.debug("heartbeat_received", subsystem=hb.subsystem, seq=hb.sequence)
            except Exception:
                break   # queue.Empty (multiprocessing.Queue raises queue.Empty)

        # --- 2. drain and dispatch fault events ---
        while True:
            try:
                event: FaultEventMsg = fault_queue.get_nowait()
                log.info(
                    "fault_received",
                    fault_code=event.fault_code.value,
                    subsystem=event.subsystem,
                )
                handler = FAULT_HANDLERS.get(event.fault_code)
                if handler is not None:
                    result: Optional[ModeChangeMsg] = handler(event)
                    if result is not None:
                        mode_queue.put(result)
                        log.info(
                            "mode_change_requested",
                            new_mode=result.new_mode.value,
                            requested_by=result.requested_by,
                        )
                else:
                    # Should never reach here given the startup assertion.
                    log.error("no_handler_for_fault", fault_code=event.fault_code.value)
            except Exception:
                break   # queue.Empty

        # --- 3. watchdog check ---
        now = time.monotonic()
        watchdog_entries, watchdog_faults = check_heartbeats(
            watchdog_entries, now, config.watchdog_max_miss_count
        )
        for wf in watchdog_faults:
            handler = FAULT_HANDLERS.get(wf.fault_code)
            if handler is not None:
                result = handler(wf)
                if result is not None:
                    mode_queue.put(result)

        # --- 4. sleep for remainder of interval ---
        elapsed = time.monotonic() - loop_start
        sleep_s = max(0.0, config.watchdog_interval_s - elapsed)
        if sleep_s > 0:
            time.sleep(sleep_s)
