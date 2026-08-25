# flight.libs.types

**Source:** `packages/flight/src/flight/libs/types/`
**Kind:** package

## Purpose

The types package holds enumerations, the `Result` error wrapper, and raw-frame value types. It
is the dependency root for most flight code.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`enums`](types/enums.md) | module | System, gimbal, fault, message, and command enumerations |
| [`result`](types/result.md) | module | `Ok`, `Err`, and `Result[T, E]` types |
| [`frames`](types/frames.md) | module | `MosaicFrame` raw sensor frame value type |

## Package interface

`flight.libs.types` re-exports:

| Name | Kind |
| --- | --- |
| `AckStatus`, `Band`, `CommandId`, `DownlinkPriority`, `FaultCode`, `FrameUsabilityTag` | enum |
| `GimbalCommandMode`, `GimbalState`, `LaunchLockState`, `LinkState`, `MessageType` | enum |
| `ModelDeployState`, `ParamKind`, `SystemMode` | enum |
| `Err`, `Ok`, `Result` | type |
| `MosaicFrame` | class |

## Interactions

None. This package is pure library code with no bus use.

## Constraints

- No other flight module is imported here.
- `Ok` and `Err` are frozen dataclasses without slots. The explicit `Generic` and `Union`
  forms are the stable public contract.
- `MosaicFrame` is not a bus message. The sensor driver passes it by direct call to the
  payload app.
- `DownlinkPriority` uses integer values `0..3`. Lower value means higher downlink priority.

## Related documents

- [`flight.libs`](../libs.md)
- [`flight.libs.types.enums`](types/enums.md)
- [`flight.libs.types.result`](types/result.md)
