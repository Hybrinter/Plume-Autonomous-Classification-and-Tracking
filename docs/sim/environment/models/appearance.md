# sim.environment.models.appearance

**Source:** `packages/sim/src/sim/environment/models/appearance.py`
**Kind:** module

## Purpose

The appearance module supplies the `AppearanceModel` Protocol and
`OracleMask`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `AppearanceModel` | Protocol | `render_feed(...) -> DriverFeed` |
| `OracleMask` | class | 50-by-50 unit mask at the projected CoG |

## Inputs and outputs

**`AppearanceModel.render_feed(geom, shutter, camera, rng) -> DriverFeed`**

- Inputs: `SceneGeometry`, `ShutterPose`, `CameraGeometry`, numpy Generator.
- Output: optional mosaic and optional mask.

## Behavior

1. `OracleMask` allocates a zero band-plane mask of camera height and width.
2. When a centroid exists, it paints a 50-by-50 square of ones around it.
3. Each edge of the square clips to the image independently.
4. A wholly off-frame centroid leaves the mask at zero.
5. `mosaic` stays `None`.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

A `None` mosaic or mask means a SIL bind must not push that slot.

## Related documents

- [`sim.environment.models`](../models.md)
- [`sim.environment.records`](../records.md)
- [`sim.environment.evaluate`](../evaluate.md)
