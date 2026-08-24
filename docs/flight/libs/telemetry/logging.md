# flight.libs.telemetry.logging

**Source:** `packages/flight/src/flight/libs/telemetry/logging.py`
**Kind:** module

## Purpose

The module configures structlog for flight and returns subsystem-bound loggers.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `configure_logging` | function | One-time structlog setup |
| `get_logger` | function | Return a logger bound to a subsystem name |

## Inputs and outputs

| Entry point | Inputs | Outputs |
| --- | --- | --- |
| `configure_logging(flight_mode)` | `bool` flight vs dev rendering | None |
| `get_logger(subsystem)` | Subsystem name string | `FilteringBoundLogger` with `subsystem` bound |

## Behavior

1. `configure_logging` selects JSON rendering when `flight_mode` is True.
2. `configure_logging` selects colorized console rendering when `flight_mode` is False.
3. Processors add log level, ISO UTC timestamp, stack info, and exception formatting.
4. `get_logger(subsystem)` binds the subsystem field on every subsequent log entry.
5. Callers pass the event name as the first positional argument to each log call.

## Errors and faults

None.

## Messages

None. Structured downlink events use `TelemetryEventMsg` separately.

## Configuration

The composition root passes `flight_mode` at startup. There is no TOML field in libs.

## Constraints

- Call `configure_logging()` exactly once before any `get_logger()` call.
- Reconfigures global structlog state.
- JSON mode emits one object per line for downlink parsing.
- Every log entry carries a `subsystem` field.

## Related documents

- [`flight.libs.telemetry`](../telemetry.md)
