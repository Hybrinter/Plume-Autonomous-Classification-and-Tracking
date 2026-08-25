# flight.libs.time

**Source:** `packages/flight/src/flight/libs/time/`
**Kind:** package

## Purpose

The time package provides an injectable clock abstraction. It separates monotonic time from
wall-clock ISO stamps.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`clock`](time/clock.md) | module | `Clock` protocol, `RealClock`, and `ManualClock` |

## Package interface

`flight.libs.time` re-exports:

| Name | Kind |
| --- | --- |
| `Clock` | Protocol |
| `RealClock` | class |
| `ManualClock` | class |

## Interactions

The composition root constructs a `Clock` and injects it into app shells. Pure cores receive
time as `now: float` arguments. They do not read a clock.

## Constraints

- Use `monotonic_s()` for intervals, timeouts, and rate limits.
- Use `wall_clock_iso()` for message timestamp fields.
- Do not mix the two channels.
- `ManualClock.advance()` moves monotonic time explicitly in tests.

## Related documents

- [`flight.libs`](../libs.md)
- [`flight.libs.time.clock`](time/clock.md)
