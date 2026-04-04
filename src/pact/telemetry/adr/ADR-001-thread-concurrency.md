# ADR-001: threading.Thread Concurrency for Telemetry Reporter

**Status:** Accepted
**Date:** 2026-04-03
**Req IDs:** REQ-OPER-HIGH-001, REQ-COMM-HIGH-001

## Context

The telemetry subsystem receives `TelemetryEventMsg` values from all other subsystems,
aggregates them into `SystemHealthSnapshot` records, and formats them as CCSDS telemetry
packets for insertion into the downlink queue. It also periodically samples system health
(mode, gimbal state, fault set, counters) to produce the `SystemHealthSnapshot`.

This is purely I/O-bound work: queue reads, struct serialization, and `CcsdsPacket` encoding.
No CPU-heavy loops. The serialization operations (`struct.pack`, JSON encoding) are fast
compared to queue wait times.

Three concurrency options were considered:
- **`multiprocessing.Process`** — overkill for an I/O-bound aggregator; adds queue
  serialization overhead on every `TelemetryEventMsg`.
- **`threading.Thread`** — appropriate for I/O-bound work; shares GIL without issue.
- **`asyncio`** — would require a bridge to `multiprocessing.Queue` (same as comms).
  Not justified here since there is no I/O multiplexing requirement.

## Decision

Telemetry runs as a `threading.Thread` within a dedicated telemetry process. The thread
reads from `queue.Queue[TelemetryEventMsg]` (internal) fed by a `multiprocessing.Queue`
bridge from other processes, formats CCSDS packets, and puts `DownlinkItemMsg` values on
the downlink queue with `DownlinkPriority.HEALTH_TELEMETRY` (highest priority).

## Consequences

### Positive
- Simple implementation: a tight `while not shutdown_event.is_set()` loop reading from
  the queue and formatting packets.
- Health telemetry is guaranteed to be placed on the downlink queue at highest priority,
  satisfying the downlink priority ordering (health > science > imagery).
- No async machinery needed; easier to reason about timing guarantees.

### Negative / Trade-offs
- A burst of `TelemetryEventMsg` values (e.g., rapid state transitions during fault recovery)
  could cause the telemetry thread to lag. Mitigation: the `TelemetryEventMsg` queue should
  have a generous `maxsize` (e.g., 256), and events can be batched into a single
  `SystemHealthSnapshot` if the queue depth exceeds a threshold.
- Thread shares the GIL with the multiprocessing Queue bridge polling loop. If bridge polling
  is implemented as a busy loop, it could starve the formatter thread. Use `blocking=True`
  with a timeout on Queue.get() instead.
