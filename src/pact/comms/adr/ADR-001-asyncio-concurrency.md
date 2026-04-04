# ADR-001: asyncio for I/O-Multiplexed Comms

**Status:** Accepted
**Date:** 2026-04-03
**Req IDs:** REQ-COMM-HIGH-001, REQ-COMM-HIGH-002, REQ-COMM-HIGH-003, GOAL-004, GOAL-008

## Context

The comms subsystem must simultaneously:
- Drain the downlink priority queue and pace output to ≤5 Mbps.
- Assemble incoming uplink chunks (model upload) across multiple packets.
- Enforce the daily byte budget (1 GB down / 100 MB up).
- Check the comm window schedule (weekdays only) on every dequeue attempt.
- Detect and handle comm timeout faults.

These are all I/O-multiplexed concerns: waiting on network sockets, timers, and queue events.
There are no CPU-heavy loops. The TDRSS radio interface is stubbed to a file or socket for
Phase I.

Two concurrency options were considered:
1. **`threading.Thread`** — simpler, but requires explicit locking for shared budget counters
   and the priority queue.
2. **`asyncio`** — cooperative multitasking; single-threaded event loop with `asyncio.Queue`
   for message passing. No locks needed for shared state within the event loop.

## Decision

The comms process uses `asyncio` as its concurrency primitive. The process entry point
(`run_comms_process`) starts an `asyncio.run()` event loop. All internal components (downlink
drainer, uplink assembler, window scheduler, budget tracker) are `async` coroutines.

Messages crossing the process boundary (from `ops/main.py`) arrive via `multiprocessing.Queue`.
A bridge coroutine polls the `multiprocessing.Queue` non-blockingly and feeds items into
`asyncio.Queue` instances within the event loop.

## Consequences

### Positive
- No locks needed for shared state (budget counters, session state) — single-threaded event
  loop serializes all access.
- Uplink chunk assembly is naturally expressed as an `async` accumulator coroutine.
- Downlink pacing (sleep between sends) is `asyncio.sleep()` — non-blocking, no thread wasted.
- `asyncio.Queue` provides built-in backpressure for the downlink queue.
- Easy to add future I/O waiters (e.g., a real TDRSS socket) without architectural changes.

### Negative / Trade-offs
- `multiprocessing.Queue` → `asyncio.Queue` bridge requires a polling loop with a small sleep
  (`asyncio.sleep(0.01)`) to avoid busy-waiting. This adds up to 10 ms latency on message
  ingestion from other processes.
- `asyncio` is less familiar than `threading` to embedded/systems developers who may maintain
  this code. All comms coroutines must be documented with `async def` rationale.
- If a blocking call accidentally enters the event loop (e.g., a synchronous file write), it
  stalls all comms coroutines. All file I/O within the comms process must use
  `asyncio.to_thread()` or be pre-validated as non-blocking.
- Integration testing requires `pytest-asyncio` or `asyncio.run()` wrappers in test fixtures.
