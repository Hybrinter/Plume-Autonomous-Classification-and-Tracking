# flight.libs.types.frames

**Source:** `packages/flight/src/flight/libs/types/frames.py`
**Kind:** pure module

## Purpose

The module defines `MosaicFrame`, the raw sensor frame value type passed from the imaging HAL
to the payload app. It is not a bus message.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `MosaicFrame` | class | Frozen raw RGB planes plus capture metadata |

## Inputs and outputs

| Entry point | Inputs | Outputs |
| --- | --- | --- |
| `MosaicFrame(...)` | `timestamp_utc`, `timestamp_s`, `frame_id`, `planes`, `exposure_us`, `gain_db`, optional per-band exposure and gain | Frozen `MosaicFrame` instance |

Fields:

| Field | Type | Description |
| --- | --- | --- |
| `timestamp_utc` | `str` | ISO 8601 capture time with millisecond precision |
| `timestamp_s` | `float` | Monotonic shutter time in seconds |
| `frame_id` | `int` | Monotonic uint32 frame counter from the driver |
| `planes` | `object` | `np.ndarray[uint16, (3, H, W)]` in `BAND_ORDER` |
| `exposure_us_by_band` | `tuple` or `None` | Per-channel exposure. `None` uses `exposure_us` |
| `gain_db_by_band` | `tuple` or `None` | Per-channel gain. `None` uses `gain_db` |
| `exposure_us` | `float` | Exposure time in microseconds |
| `gain_db` | `float` | Analogue gain in dB |

## Behavior

1. The sensor driver constructs a `MosaicFrame` after each capture.
2. The driver passes the frame by direct call into `PayloadApp.process_frame()`.
3. The payload preprocessing pipeline reads the RGB planes and metadata.
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
- The `planes` field is typed `object` to avoid a numpy import at the libs layer.
- Callers retrieve the array with `np.asarray(frame.planes)`.
- Preprocessing runs inside the payload app after construction.

## Related documents

- [`flight.libs.types`](../types.md)
- [`flight.libs.types.enums`](enums.md)
