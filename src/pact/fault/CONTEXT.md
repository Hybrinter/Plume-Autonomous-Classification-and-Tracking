# fault/ -- Agent Context

## Purpose

Heartbeat watchdog that monitors all subsystem processes, dispatches faults to registered
handlers, and manages safe-mode entry. The fault process is the system's last line of
defense against undetected process failure.

## Defining Design Decision

Runs as `multiprocessing.Process`, not a thread. The inference GPU kernel can hold the
GIL for hundreds of milliseconds. A thread-based watchdog would fail to fire during that
window. A separate OS process is GIL-immune and will always receive heartbeats on schedule.

## Invariants

- Every `FaultCode` member must have a registered handler in `FAULT_HANDLERS`. Absence is
  asserted at startup -- a missing handler crashes the fault process immediately, before
  any faults can be missed.
- Safe mode is sticky: `exit_safe_mode()` exists but is only callable via an explicit
  ground command. There is no automatic recovery in Phase I.

## Gotchas

`check_thermal()` and `check_power()` in `detector.py` are called each iteration with
mocked `0.0` sensor values. Both detectors are structurally wired but functionally blind.
A system showing no thermal or power faults is not evidence of actual health -- it is
evidence that real sensor readings are not yet connected.

## Phase II Gaps

- Thermal and power detectors need real Xavier sensor readings (power draw as thermal
  proxy via INA3221).
- Safe-mode exit via ground command is not implemented.
