# flight.payload.gimbal.scene

**Source:** `packages/flight/src/flight/payload/gimbal/scene.py`
**Kind:** pure module

## Purpose

This module selects the Earth point for one outer tick and decides when a new
target identity cold-starts the residual filter.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SceneSource` | enum | `COG`, `BORESIGHT`, or `NONE` |
| `SceneEstimate` | dataclass | Prediction time, source, optional ECEF point, optional `LosPrediction`, and navigation validity |
| `select_scene` | function | Selects the scene point and predicts co-rotating rates |
| `acquire_resets_residual` | function | True when TRACKING acquire must cold-start the residual |

## Inputs and outputs

`select_scene` takes arbiter mode, an optional stored CoG, optional ISS ECI
position, velocity, and UTC, encoder elevation, height-proxy offset, and Earth
rotation constants. It returns a `SceneEstimate`.

`acquire_resets_residual` takes previous and new arbiter mode, previous aggregate
liveness, and previous versus new `blob_id` sets. It returns a boolean.

Prediction time `t_utc_s` is ISS UTC when navigation is present. It is not outer
monotonic now.

## Behavior

1. SAFE returns source `NONE` and no Earth point.
2. REWIND returns source `BORESIGHT`. With valid ISS, it intersects the current
   boresight with the height-proxy ellipsoid. That hit is scene rate only. The
   function does not treat it as a CoG.
3. TRACKING with a stored CoG returns source `COG` and predicts from that point.
   TRACKING with no CoG returns source `NONE`.
4. Missing ISS sets `nav_valid` to false and leaves `los` empty. A valid
   `LosPrediction` may still hold a 0.0 rate for a stationary scene.
5. `acquire_resets_residual` is true for a TRACKING blob from a cold aggregate,
   a blob that leaves REWIND, or a TRACKING blob set with no overlapping
   `blob_id`. An empty frame does not reset.

## Errors and faults

None.

## Messages

None.

## Configuration

Height-proxy offset comes from `predictor.cog_height_m`. WGS-84 scalars and Earth
rate come from `EphemerisConfig`. Blob identity uses IDs assigned by
`match_blobs` at `vision.blob_iou_match_threshold`.

## Constraints

The module is pure. Callers must not write a boresight hit into
`r_cog_ecef_m`. Unknown navigation is `nav_valid` false, not omega 0.0.

## Related documents

- [`flight.payload.gimbal.intersect`](intersect.md)
- [`flight.payload.gimbal.predictor`](predictor.md)
- [`flight.payload.gimbal.outer`](outer.md)
- [`flight.payload.gimbal.arbiter`](arbiter.md)
- [`flight.payload.control`](../control.md)
