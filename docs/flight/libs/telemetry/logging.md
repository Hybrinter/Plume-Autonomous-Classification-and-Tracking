# flight.libs.telemetry.logging

**Source:** `packages/flight/src/flight/libs/telemetry/logging.py`
**Kind:** module

## Purpose

This module configures structlog for flight and development modes. It binds a subsystem
name to each logger.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `configure_logging` | function | Set global structlog processors and renderer |
| `get_logger` | function | Return a logger bound to a subsystem name |

## Inputs and outputs

- `configure_logging(flight_mode: bool) -> None`
- `get_logger(subsystem: str) -> FilteringBoundLogger`

## Behavior

1. `configure_logging` selects a renderer from `flight_mode`.
2. When `flight_mode` is true, each entry renders as one JSON object per line.
3. When `flight_mode` is false, entries use the colorized console renderer.
4. Processors include context merge, log level, ISO UTC timestamp, stack info, and exception
   formatting.
5. `get_logger` binds the given `subsystem` string to the structlog logger.

## Errors and faults

None.

## Messages

None.

## Configuration

`configure_logging` takes a boolean `flight_mode` flag at startup. It does not read TOML.

## Constraints

- `configure_logging` reconfigures global structlog state.
- Call it exactly once before obtaining loggers.
- The `event` field is the first positional argument on each log call (structlog convention).

## Related documents

- [`flight.libs.telemetry`](flight/libs/telemetry.md)
