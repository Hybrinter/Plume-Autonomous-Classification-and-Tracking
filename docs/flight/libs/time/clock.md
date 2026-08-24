# flight.libs.time.clock

**Source:** `packages/flight/src/flight/libs/time/clock.py`
**Kind:** module

## Purpose

This module defines the injectable clock protocol and two implementations. Monotonic time
supports control intervals. Wall-clock ISO strings support message stamps.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Clock` | protocol | `monotonic_s()` and `wall_clock_iso()` |
| `RealClock` | class | Production clock using system time |
| `ManualClock` | class | Test clock with explicit time control |

### `Clock`

| Name | Kind | Description |
| --- | --- | --- |
| `monotonic_s` | method | Monotonic seconds for intervals and timeouts |
| `wall_clock_iso` | method | UTC ISO 8601 string with millisecond precision |

### `ManualClock`

| Name | Kind | Description |
| --- | --- | --- |
| `advance` | method | Add seconds to monotonic time |
| `set_wall_clock` | method | Set the returned wall-clock ISO string |

## Inputs and outputs

- `RealClock()` needs no arguments.
- `ManualClock(monotonic_s=0.0, wall_clock="2026-01-01T00:00:00.000Z")` sets initial values.
- `monotonic_s() -> float`
- `wall_clock_iso() -> str` returns `YYYY-MM-DDTHH:MM:SS.mmmZ`.

## Behavior

1. `RealClock.monotonic_s` returns `time.monotonic()`.
2. `RealClock.wall_clock_iso` returns the current UTC time in millisecond ISO form with a
   `Z` suffix.
3. `ManualClock` stores monotonic and wall-clock values set at construction.
4. `ManualClock.advance(delta_s)` increases stored monotonic seconds by `delta_s`.
5. `ManualClock.set_wall_clock(wall_clock)` replaces the stored ISO string.

## Errors and faults

None.

## Messages

None.

## Configuration

The composition root selects `RealClock` or `ManualClock` from `EnvironmentConfig.clock`.
This module does not read config files.

## Constraints

- `Clock` is `@runtime_checkable`.
- Pure logic receives time as arguments; it does not call a clock internally.
- Wall-clock format matches `utc_now_iso()` in `flight.libs.messages`.

## Related documents

- [`flight.libs.time`](flight/libs/time.md)
- [`flight.libs.messages`](flight/libs/messages.md)
