# flight.libs.types.frames

**Source:** `packages/flight/src/flight/libs/types/frames.py`
**Kind:** pure module

## Purpose

The module defines `MosaicFrame`, the raw sensor frame value type passed from the imaging HAL
to the payload app. It is not a bus message.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `MosaicFrame` | class | Frozen raw CFA mosaic plane plus capture metadata |

## Inputs and outputs

| Entry point | Inputs | Outputs |
| --- | --- | --- |
| `MosaicFrame(...)` | `timestamp_utc`, `frame_id`, `mosaic`, `exposure_us`, `gain_db`, optional `capture_monotonic_s` | Frozen `MosaicFrame` instance |

Fields:

| Field | Type | Description |
| --- | --- | --- |
| `timestamp_utc` | `str` | ISO 8601 capture time with millisecond precision |
| `frame_id` | `int` | Monotonic uint32 frame counter from the driver |
| `mosaic` | `object` | `np.ndarray[uint16, (H, W)]` raw 2x2-CFA mosaic plane |
| `exposure_us` | `float` | Exposure time in microseconds |
| `gain_db` | `float` | Analogue gain in dB |
| `capture_monotonic_s` | `float` | Monotonic shutter time in seconds (0 = app uses loop `now`) |

## Behavior

1. The sensor driver constructs a `MosaicFrame` after each capture.
2. The driver passes the frame by direct call into `PayloadApp.process_frame()`.
3. The payload preprocessing pipeline reads the mosaic array and metadata.
4. The frame is immutable after construction.

## Errors and faults

None at construction. Downstream preprocessing may emit `FaultCode.FRAME_MALFORMED` or
`FaultCode.CALIBRATION_INVALID`.

## Messages

None. Raw frames never ride the bus.

## Configuration

None.

## Constraints

- `MosaicFrame` is not a bus message. Large arrays stay off the bus.
- The `mosaic` field is typed `object` to avoid a numpy import at the libs layer.
- Callers retrieve the array with `np.asarray(frame.mosaic)`.
- Preprocessing runs inside the payload app after construction.

## Related documents

- [`flight.libs.types`](../types.md)
- [`flight.libs.types.enums`](enums.md)
