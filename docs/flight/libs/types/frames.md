# flight.libs.types.frames

**Source:** `packages/flight/src/flight/libs/types/frames.py`
**Kind:** pure module

## Purpose

This module defines raw-frame value types exchanged between the imaging HAL and the payload
app. These types are not bus messages.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `MosaicFrame` | dataclass | Raw CFA mosaic plane plus capture metadata |

### `MosaicFrame` fields

| Field | Type | Description |
| --- | --- | --- |
| `timestamp_utc` | `str` | ISO 8601 capture time, millisecond precision |
| `frame_id` | `int` | Monotonic uint32 frame counter from the driver |
| `mosaic` | `object` | `np.ndarray[uint16, (H, W)]` raw 2x2 CFA plane |
| `exposure_us` | `float` | Exposure time in microseconds |
| `gain_db` | `float` | Analog gain in dB |

## Inputs and outputs

Construct `MosaicFrame(timestamp_utc, frame_id, mosaic, exposure_us, gain_db)`. The
dataclass is frozen after construction.

## Behavior

1. The sensor driver builds a `MosaicFrame` for each captured frame.
2. The driver passes the frame to the payload app by direct function call.
3. The payload preprocessing pipeline consumes the mosaic and metadata.
4. Callers obtain the array with `np.asarray(frame.mosaic)`.

## Errors and faults

None.

## Messages

None. Raw frames do not publish to the message bus.

## Configuration

None.

## Constraints

- `mosaic` is typed `object` to avoid a numpy import in `flight.libs.types`.
- The frame is immutable after construction.
- Preprocessing runs in the payload app after the driver delivers the frame.

## Related documents

- [`flight.libs.types`](flight/libs/types.md)
- [`flight.payload`](flight/payload.md)
