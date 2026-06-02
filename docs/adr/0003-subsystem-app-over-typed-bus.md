# ADR 0003: Subsystem-app model over a typed message bus

**Status:** Accepted (2026-05-30)

## Context

The legacy structure spread one logical pipeline across many `multiprocessing` processes and
threads wired by ~11 named queues created in `ops/main.py`. The topology was hard to see, the queue
plumbing was duplicated, and pure logic was entangled with transport.

## Decision

Model each subsystem as an isolated **app**: a thin imperative shell around a pure decision core,
communicating with other apps **only** over a typed pub/sub `MessageBus` (routed by exact message
type). Peer apps never cross-import. A single **composition root** (`flight.core`, and `sim.sil`
for SIL) owns the bus, the clock, the drivers, and the scheduler, and wires everything via one
driver-agnostic `build_apps()`. Transport is in-process `queue.Queue` today; a multiprocessing
transport can replace the queue factory later without touching app code.

## Consequences

- The full topology is visible in one place (`build_apps`); adding a channel means adding a message
  type, not threading a new queue through call sites.
- The same `build_apps` runs in flight (real drivers) and in the deterministic SIL (sim drivers),
  so the integration test exercises the real wiring.
- Large artifacts (tensors, masks) must stay off the bus (serialization cost); the bus carries
  compact records, preserving the preprocessing co-location discipline.
- Standard envelopes (`CommandMsg`, `TelemetryEventMsg`, fault/heartbeat/mode events) give the
  "everything commandable, everything telemetered" property.
