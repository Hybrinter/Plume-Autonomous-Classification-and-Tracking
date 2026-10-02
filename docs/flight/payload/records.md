# flight.payload.records

**Source:** `packages/flight/src/flight/payload/records.py`
**Kind:** pure module

## Purpose

The module holds compact payload observation and activation-context value
records shared between the shell and the pure cores.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `VisionSample` | dataclass | Frame ID, shutter time, error, centroid, exposure, blobs, and ISS |
| `IssSample` | dataclass | ISS ECI state for the predictor |
| `ActivationKey` | dataclass | Session epoch plus activation sequence |
| `CaptureContext` | dataclass | Activation key, policy revision, and model version |
| `CapturedVision` | dataclass | Vision sample tagged with its capture context |
| `HealthSample` | dataclass | Feedback validity, inhibit confirmation, and containment flags |

## Inputs and outputs

All records are frozen slots dataclasses. `VisionSample.theta_g_rad` defaults
to `None`. `CaptureContext`, `CapturedVision`, `ActivationKey`, and
`HealthSample` fields carry data only; nothing here executes behavior.

## Behavior

The app shell constructs `IssSample` from ephemeris reads and `VisionSample`
from gated inference results. Pure cores consume them as tick inputs.
`CapturedVision` tags a vision sample with the activation key, policy
revision, and model identity under which its capture ran.

## Errors and faults

None.

## Messages

None. These records are call/return values, not bus messages.

## Configuration

None.

## Constraints

The records are pure data. They never perform I/O, read clocks, or touch the
bus.

## Related documents

- [`flight.payload.control`](control.md)
- [`flight.payload.app`](app.md)
- [`flight.payload.graphs.base`](graphs/base.md)
