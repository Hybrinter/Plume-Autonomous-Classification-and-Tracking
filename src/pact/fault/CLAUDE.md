# Fault Detection Subsystem — `pact/fault/`

## Purpose
Monitor all processes via heartbeat, detect faults, invoke handlers, manage safe mode.

## Satisfies
- REQ-SAFE-HIGH-002 — automatic safe mode entry on unhandled fault
- REQ-GIMB-HIGH-003 — gimbal runaway detection and safe mode entry
- GOAL-006 — fault-tolerant operations with safe mode recovery path

## Owns
- `ModeChangeMsg` — produces safe mode transition messages consumed by ops/main.py
- `FaultEventMsg` — re-emits enriched fault events (with handler result) for logging

## Consumes
- `HeartbeatMsg` — from all subsystems, used by watchdog to detect process death
- `FaultEventMsg` — from all subsystems, dispatched to FAULT_HANDLERS

## Key Invariants
- `FAULT_HANDLERS` covers every `FaultCode` member — verified at startup in
  `run_fault_process()` via an assertion. A missing handler is a programming error
  caught immediately, not silently at fault time.
- Safe mode is entered immediately on any unhandled fault (i.e., a FaultCode with no
  registered handler, or a handler that returns a ModeChangeMsg to SAFE).
- The fault process runs in its own OS process, immune to GIL starvation caused by
  the inference or preprocessing subsystems. See `fault/adr/ADR-001`.
- `run_fault_process()` accepts a `stop_event: multiprocessing.Event` argument — it exits
  cleanly when the event is set, allowing graceful shutdown.
- `check_thermal(temp_c, cfg) -> Optional[FaultEventMsg]` and
  `check_power(watts, cfg) -> Optional[FaultEventMsg]` are implemented in `detector.py`.
  They return `None` if below threshold, or a `FaultEventMsg` to emit directly. Both are
  called each process iteration with mocked sensor values until hardware telemetry is
  available.

## Concurrency
`multiprocessing.Process` + `multiprocessing.Queue` — see `fault/adr/ADR-001`.

Rationale: the fault monitor must be immune to GIL starvation. Running as a separate
OS process ensures the watchdog timer fires even when the inference process is running
a long GPU kernel.

## Known Gaps / TODOs
- No automatic safe-mode exit — requires a ground command (out of scope Phase I). The
  `exit_safe_mode()` function exists but is only called when the ops process receives
  an explicit ground command (not yet implemented).
- Fault detection logic for thermal (`check_thermal`) and power (`check_power`) is
  implemented in `detector.py`, but both are invoked with mocked constant values (0.0)
  since no hardware sensor interface exists in Phase I. Real sensor readings require a
  hardware abstraction layer (Phase II).
