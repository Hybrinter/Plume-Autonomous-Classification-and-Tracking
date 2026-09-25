# flight.payload.preprocess.stack

**Source:** `packages/flight/src/flight/payload/preprocess/stack.py`
**Kind:** pure module

## Purpose

This module stacks a prism camera buffer into channel-major planes. The AP-3200T-USB
delivers three registered sensors. A 2-D buffer is malformed.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `stack_channels` | function | Returns float32 `(3, H, W)` from `(3, H, W)` or `(H, W, 3)` |

## Inputs and outputs

`stack_channels(buffer)` returns `Result[np.ndarray, FaultCode]`. Success is float32
shape `(3, H, W)`.

## Behavior

1. Accept a channel-major buffer whose first axis has length 3.
2. Accept a channel-last buffer whose last axis has length 3 and move that axis first.
3. Cast the result to float32.
4. Reject every other rank or channel count.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(FRAME_MALFORMED)` | Buffer is not a 3-channel volume |

## Messages

None.

## Configuration

None. Channel names are applied later by `select_bands` from `SensorConfig.channel_layout`.

## Constraints

The function checks rank and channel count. It does not check absolute height and width.
Callers pass the raw camera buffer once. Downstream stages see only the stack.

## Related documents

- [`flight.payload.preprocess`](../preprocess.md)
- [`flight.payload.preprocess.radiometric`](radiometric.md)
