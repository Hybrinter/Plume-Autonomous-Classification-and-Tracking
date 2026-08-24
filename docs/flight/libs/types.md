# flight.libs.types

**Source:** `packages/flight/src/flight/libs/types`
**Kind:** package

## Purpose

This package holds pure flight types: enumerations, the `Result` wrapper, and raw-frame
value types. It sits at the bottom of the flight import graph.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`enums`](flight/libs/types/enums.md) | module | System, fault, message, and command enumerations |
| [`frames`](flight/libs/types/frames.md) | module | `MosaicFrame` raw sensor frame type |
| [`result`](flight/libs/types/result.md) | module | `Ok`, `Err`, and `Result` type alias |

## Package interface

Re-exports:

- Enumerations: `AckStatus`, `Band`, `CommandId`, `DownlinkPriority`, `FaultCode`,
  `FrameUsabilityTag`, `GimbalCommandMode`, `GimbalState`, `LaunchLockState`, `LinkState`,
  `MessageType`, `ModelDeployState`, `ParamKind`, `SystemMode`
- Result types: `Ok`, `Err`, `Result`
- Frame types: `MosaicFrame`

## Interactions

`flight.libs.messages`, `flight.libs.commands`, `flight.libs.ccsds`, HAL drivers, and
subsystem apps import types from `flight.libs.types`. `MosaicFrame` passes from the sensor
driver to the payload app by direct call, not on the bus.

## Constraints

- Import from `flight.libs.types`, not from inner modules.
- This package imports no other flight modules.
- Enum string values equal member names except `DownlinkPriority`, which uses integers 0..3.

## Related documents

- [`flight.libs`](flight/libs.md)
- [`flight.libs.types.enums`](flight/libs/types/enums.md)
- [`flight.libs.types.frames`](flight/libs/types/frames.md)
- [`flight.libs.types.result`](flight/libs/types/result.md)
