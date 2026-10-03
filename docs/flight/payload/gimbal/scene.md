# flight.payload.gimbal.scene

**Source:** `packages/flight/src/flight/payload/gimbal/scene.py`
**Kind:** pure module

## Purpose

This module predicts the scene Earth point for one outer tick from either a
stored CoG or the boresight height-proxy intersect. The caller selects the
entry point; the module carries no mode dispatch.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SceneSource` | enum | `COG`, `BORESIGHT`, or `NONE` |
| `SceneEstimate` | dataclass | Prediction time, source, optional ECEF point, optional `LosPrediction`, and navigation validity |
| `cog_scene` | function | Predicts co-rotating rates from the stored target CoG |
| `boresight_scene` | function | Predicts co-rotating rates from the boresight height-proxy intersect |

## Inputs and outputs

`cog_scene` takes an optional stored CoG, optional ISS ECI position, velocity,
and UTC, and Earth rotation constants. `boresight_scene` takes the optional ISS
state, encoder elevation, height-proxy offset, Earth rotation constants, and
WGS-84 scalars. Both return a `SceneEstimate`.

Prediction time `t_utc_s` is ISS UTC when navigation is present. It is not outer
monotonic now.

## Behavior

1. `cog_scene` returns source `COG` and the stored point when a CoG exists, and
   source `NONE` when no CoG exists.
2. `boresight_scene` returns source `BORESIGHT` always. With valid ISS it
   intersects the current boresight with the height-proxy ellipsoid. That hit
   is scene rate only. The function does not treat it as a CoG. A missed ray
   or absent ISS leaves the point empty.
3. Missing or partial ISS sets `nav_valid` to false and leaves `los` empty. A
   valid `LosPrediction` may still hold a 0.0 rate for a stationary scene.

## Errors and faults

None.

## Messages

None.

## Configuration

Height-proxy offset comes from `predictor.cog_height_m`. WGS-84 scalars and
Earth rate come from `EphemerisConfig`.

## Constraints

The module is pure. Callers must not write a boresight hit into
`r_cog_ecef_m`. Unknown navigation is `nav_valid` false, not omega 0.0.

## Related documents

- [`flight.payload.gimbal.intersect`](intersect.md)
- [`flight.payload.gimbal.predictor`](predictor.md)
- [`flight.payload.gimbal.outer`](outer.md)
- [`flight.payload.graphs.operate`](../graphs/operate.md)
- [`flight.payload.control`](../control.md)
