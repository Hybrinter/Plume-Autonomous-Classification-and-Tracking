# ADR 0006: ISS-attached reliability posture (fail-safe / ground-recoverable)

**Status:** Accepted (2026-05-30)

## Context

PACT is an **ISS-attached external payload**, not a free-flying satellite and not related to any
particular operator's bus. The station provides the services a free-flyer must implement itself --
primary power, thermal sink, attitude, and the downlink path to the ground -- and the payload is
continuously commandable and recoverable through the station. This materially lowers the software
reliability burden compared with an autonomous free-flyer, while NASA payload safety requirements
still bound hazardous functions.

## Decision

Target a **fail-safe / ground-recoverable / graceful-degradation** posture rather than full
autonomous fault recovery:

- On a detected fault, enter **SAFE** (minimal activity) and wait for ground command, rather than
  attempting heavy autonomous recovery. SAFE exit is an explicit ground command.
- Each producing subsystem **self-reports** its faults onto the bus; the FDIR app routes
  SAFE-triggering faults to a `ModeChangeMsg(SAFE)` and runs a heartbeat watchdog for silent death.
- Implement **no RF/CCSDS stack** -- the station owns the RF link; `iss_iface` is a thin bridge
  behind a `StationLink` Protocol whose exact wire protocol is deferred to the avionics interface.
- Still honor NASA payload safety inhibits for hazardous functions (gimbal stored energy,
  batteries, RF).

## Consequences

- FDIR is pragmatic: detect, isolate to SAFE, and rely on the station/ground for recovery -- not a
  redundant autonomous-recovery system.
- The legacy CCSDS/TDRSS/budget logic is out of scope; downlink is handing items to the station.
- The reliability relaxation is explicitly tied to the ISS context and must be revisited if the
  payload ever flies in a different configuration.
