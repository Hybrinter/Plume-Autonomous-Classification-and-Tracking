# flight.libs.time.clock

**Source:** `packages/flight/src/flight/libs/time/clock.py`
**Kind:** module

## Purpose

The module defines the injectable `Clock` protocol and production and test implementations.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Clock` | Protocol | Monotonic, UTC, and wall-clock time source |
| `RealClock` | class | Production clock using system time |
| `ManualClock` | class | Deterministic test clock |

### Clock methods

| Name | Kind | Description |
| --- | --- | --- |
| `monotonic_s()` | method | Monotonic seconds for intervals and rates |
| `utc_s()` | method | UTC Unix seconds for ephemeris and Earth rotation |
| `wall_clock_iso()` | method | UTC ISO 8601 string with millisecond precision |

### ManualClock extras

| Name | Kind | Description |
| --- | --- | --- |
| `advance(delta_s)` | method | Add seconds to monotonic and UTC time |
| `set_wall_clock(wall_clock)` | method | Replace the wall-clock ISO string |

## Inputs and outputs

| Entry point | Inputs | Outputs |
| --- | --- | --- |
| `RealClock()` | None | Clock backed by `time.monotonic()` and UTC |
| `ManualClock(monotonic_s, utc_s, wall_clock)` | Initial times | Deterministic clock |
| `monotonic_s()` | None | `float` seconds |
| `utc_s()` | None | `float` Unix UTC seconds |
| `wall_clock_iso()` | None | `str` in `YYYY-MM-DDTHH:MM:SS.mmmZ` format |

## Behavior

1. `RealClock.monotonic_s()` returns `time.monotonic()`.
2. `RealClock.utc_s()` returns `time.time()`.
3. `RealClock.wall_clock_iso()` formats current UTC with a trailing `Z`.
4. `ManualClock` stores monotonic, UTC, and wall-clock values set at construction.
5. `ManualClock.advance(delta_s)` increments stored monotonic and UTC time.
6. `ManualClock.set_wall_clock()` replaces the stored ISO string.

## Errors and faults

None.

## Messages

App shells stamp messages with `clock.wall_clock_iso()` or `utc_now_iso()` from messages.

## Configuration

The composition root selects `RealClock` or `ManualClock` from `EnvironmentConfig.clock`.

## Constraints

- Pure logic does not read a clock. Time is passed in as arguments.
- Use monotonic time for control intervals. Use `utc_s()` for ephemeris. Use
  wall-clock ISO for message stamps.
- Do not use one channel for the other.
- `Clock` is `@runtime_checkable`.

## Related documents

- [`flight.libs.time`](../time.md)
- [`flight.libs.messages.messages`](../messages/messages.md)
