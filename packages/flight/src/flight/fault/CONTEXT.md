# `fault` Subsystem Context

Non-obvious, cross-cutting context for the FDIR subsystem -- not derivable from the
individual files or their docstrings.

## Functional core / imperative shell

- `watchdog` and `policy` are pure: no clock, no I/O, no bus. They take time and
  timestamps as arguments (`now` monotonic seconds, `now_iso` wall-clock string) and
  thread the `entries` dict in and out instead of mutating it. `FaultApp` (`app.py`) is
  the only place that reads the clock, touches the bus, or owns mutable state -- so the
  decision logic is replayable from logs and unit-testable without a running system.
- Two clock sources are intentionally distinct: `monotonic_s()` drives watchdog interval
  math; `wall_clock_iso()` only stamps emitted messages. Never use wall-clock for elapsed
  timing -- it is not monotonic.

## Policy replaces the legacy dispatch table

- The old per-`FaultCode` `Callable` table (`FAULT_HANDLERS`) is gone. It is replaced by
  the `SAFE_TRIGGERING_FAULTS` frozenset + `decide_mode_change()` membership test. This
  removes dynamic dispatch (a function-pointer table) in favor of a static partition, per the
  codebase's no-dynamic-dispatch typing rule (`.claude/rules/strong_typing.md`).
- The SAFE vs. log-and-continue partition is preserved from the legacy handlers, plus
  `GIMBAL_FAULT` (added 2026-06-11, ADR 0008 -- a driver-level gimbal failure). SAFE:
  `INFERENCE_NAN`, `CAMERA_STALL`, `THERMAL_OVER_LIMIT`, `POWER_OVER_LIMIT`, `GIMBAL_RUNAWAY`,
  `GIMBAL_FAULT`, `WATCHDOG_EXPIRE`, `MODEL_CORRUPT`, `PROCESS_DIED`. Log-and-continue
  (no mode change): `NONE`, `INFERENCE_TIMEOUT`, `STORAGE_FULL`, `COMM_TIMEOUT`. Edit this
  set deliberately -- it is the de-facto safety policy.

## This app does not detect faults -- it routes them

- There is intentionally no `check_thermal` / `check_power` / `detect_faults` here. Every
  producing subsystem self-reports its own faults as `FaultEventMsg`. `FaultApp` only (a)
  watches heartbeats and (b) maps already-raised faults to `ModeChangeMsg(SAFE)`. The one
  fault this subsystem originates is `WATCHDOG_EXPIRE`, which it then routes through the
  same policy as any externally raised fault.

## Watchdog gotchas

- `check_heartbeats` does NOT remove an entry after emitting a fault; misses keep
  accumulating and the fault re-fires each tick until a heartbeat resets `miss_count` to 0.
- `build_entries` seeds `last_heartbeat_time=now`, granting every subsystem one full
  interval before its first miss.
- Heartbeats from subsystems not in `monitored` are silently ignored (the `in working`
  guard in `tick`).

## SAFE exit is operator-only

- `exit_safe_mode` exists but is never called by this subsystem; leaving SAFE requires an
  explicit ground command. No automatic recovery.

## SAFE now actuates the gimbal (ADR 0008)

- `ModeChangeMsg(SAFE)` is no longer a no-op for the payload. The payload app drains mode changes
  each frame (`poll_mode_changes`) and threads `safe_commanded`/`safe_cleared` into the controller;
  on SAFE the arbiter latches and issues a STOW `GimbalRequest`, and the app has a shell-level
  fallback that stows directly if a SAFE arrives while the camera has stalled. Recovery is a ground
  `ModeChangeMsg` with a *non-SAFE* mode, which the arbiter treats as `safe_cleared`. This app
  still only *routes* faults to SAFE; the *consumption* of SAFE lives in the payload arbiter.
