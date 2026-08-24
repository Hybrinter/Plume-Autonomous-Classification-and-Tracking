# flight.libs.time

**Source:** `packages/flight/src/flight/libs/time`
**Kind:** package

## Purpose

This package provides an injectable clock abstraction. Apps and pure cores receive a
`Clock` instance. They do not read system time directly inside pure logic.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`clock`](flight/libs/time/clock.md) | module | `Clock` protocol, `RealClock`, `ManualClock` |

## Package interface

Re-exports:

- `Clock`
- `RealClock`
- `ManualClock`

## Interactions

The composition root constructs `RealClock` or `ManualClock` from `EnvironmentConfig.clock`
and injects it into apps. Pure functions receive `now: float` from `Clock.monotonic_s()`.
Message timestamps use `Clock.wall_clock_iso()` or `utc_now_iso()` from messages.

## Constraints

- Use `monotonic_s()` for intervals, timeouts, and rate limits.
- Use `wall_clock_iso()` for message timestamp strings.
- Do not mix the two channels.

## Related documents

- [`flight.libs`](flight/libs.md)
- [`flight.libs.time.clock`](flight/libs/time/clock.md)
