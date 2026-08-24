# flight.libs.telemetry

**Source:** `packages/flight/src/flight/libs/telemetry`
**Kind:** package

## Purpose

This package configures structured logging for flight processes. It wraps structlog with a
subsystem field on every log entry.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`logging`](flight/libs/telemetry/logging.md) | module | `configure_logging`, `get_logger` |

## Package interface

Re-exports:

- `configure_logging`
- `get_logger`

## Interactions

The composition root calls `configure_logging` once at startup. Subsystem apps call
`get_logger` with a subsystem name. Log entries use structlog; the first positional argument
is the event name.

## Constraints

- Call `configure_logging` once before any logger use.
- Every log entry carries a bound `subsystem` field.

## Related documents

- [`flight.libs`](flight/libs.md)
- [`flight.libs.telemetry.logging`](flight/libs/telemetry/logging.md)
