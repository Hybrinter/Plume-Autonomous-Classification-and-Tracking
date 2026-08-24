# flight.libs.version

**Source:** `packages/flight/src/flight/libs/version.py`
**Kind:** module

## Purpose

This module exposes the flight software version string.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `FLIGHT_VERSION` | constant | Semantic version string (`"0.1.0"`) |
| `flight_version` | function | Returns `FLIGHT_VERSION` |

## Inputs and outputs

- `flight_version()` takes no arguments. It returns the version string.

## Behavior

1. `FLIGHT_VERSION` holds the current package version.
2. `flight_version()` returns `FLIGHT_VERSION`.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

None.

## Related documents

- [`flight.libs`](flight/libs.md)
