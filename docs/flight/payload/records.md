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
| `CaptureContext` | dataclass | Activation key, policy revision, model version, and containment generation |
| `CapturedVision` | dataclass | Vision sample tagged with its capture context |
| `HealthSample` | dataclass | Feedback validity, inhibit confirmation, and containment flags |

## Inputs and outputs

All records are frozen slots dataclasses. `VisionSample.theta_g_rad` defaults
to `None`. `CaptureContext`, `CapturedVision`, and `HealthSample` fields
carry data only; nothing here executes behavior. `ActivationKey` is imported
from `flight.libs.types`; it is not declared in this module.

## Behavior

The app shell constructs `IssSample` from ephemeris reads and `VisionSample`
from raw inference output; the OPERATE graph applies the vision acceptance
gates when it consumes the sample. `CapturedVision` tags a vision sample with
the activation key, policy revision, containment generation, and model
identity under which its capture ran.

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
