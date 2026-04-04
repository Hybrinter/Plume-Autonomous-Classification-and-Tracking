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

## Concurrency
`multiprocessing.Process` + `multiprocessing.Queue` — see `fault/adr/ADR-001`.

Rationale: the fault monitor must be immune to GIL starvation. Running as a separate
OS process ensures the watchdog timer fires even when the inference process is running
a long GPU kernel.

## Known Gaps / TODOs
- No automatic safe-mode exit — requires a ground command (out of scope Phase I). The
  `exit_safe_mode()` function exists but is only called when the ops process receives
  an explicit ground command (not yet implemented).
- Thermal and power readings used in fault detection are stubbed; no hardware sensor
  interface is available in Phase I.
