# flight.libs.telemetry

**Source:** `packages/flight/src/flight/libs/telemetry/`
**Kind:** package

## Purpose

The telemetry package configures structured logging for flight software. It binds every log
entry to a subsystem name.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`logging`](telemetry/logging.md) | module | `configure_logging` and `get_logger` |

## Package interface

`flight.libs.telemetry` re-exports:

| Name | Kind |
| --- | --- |
| `configure_logging` | function |
| `get_logger` | function |

## Interactions

The composition root calls `configure_logging()` once at startup. Each app shell obtains a
logger with `get_logger(subsystem)`. Apps also publish `TelemetryEventMsg` on the bus for
structured downlink events.

## Constraints

- Call `configure_logging()` exactly once before any logger is obtained.
- Every log call uses the event name as the first positional argument.
- `flight_mode=True` selects JSON rendering. `flight_mode=False` selects console rendering.

## Related documents

- [`flight.libs`](../libs.md)
- [`flight.libs.telemetry.logging`](telemetry/logging.md)
