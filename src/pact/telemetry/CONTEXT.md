# telemetry/ -- Agent Context

## Purpose

Aggregates health events from all subsystems into periodic `SystemHealthSnapshot` packets,
always emitted at the highest downlink priority. Contains no fault thresholds or detection
logic -- only aggregation and formatting.

## Defining Design Decision

Downlink priority is hardcoded as `DownlinkPriority.HEALTH_TELEMETRY` inside `reporter.py`
and cannot be overridden by callers. Health telemetry must never be starved by science or
imagery data regardless of queue depth. Enforcing this in the reporter (not at call sites)
prevents accidental priority downgrade.

## Invariants

- `SystemHealthSnapshot` is frozen -- no incremental update. Each snapshot is constructed
  fresh from current state.
- This subsystem contains no fault thresholds and no detection logic. All fault decisions
  live in `fault/`.

## Gotchas

Thermal and power fields in `SystemHealthSnapshot` are always `0.0` in Phase I. Do not
interpret `thermal_c = 0.0` or `power_w = 0.0` as evidence that the system is running
cool or at low power -- the hardware sensor interface does not exist yet.

## Phase II Gaps

- Thermal and power readings need a hardware abstraction layer.
- CCSDS packet encoding uses `struct.pack` as a placeholder -- full space packet framing
  is Phase II.
