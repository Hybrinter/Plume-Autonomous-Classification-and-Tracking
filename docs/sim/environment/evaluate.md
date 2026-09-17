# sim.environment.evaluate

**Source:** `packages/sim/src/sim/environment/evaluate.py`
**Kind:** module

## Purpose

The evaluate module runs the world-model pipeline for one shutter.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EnvironmentModels` | class | Frozen bundle of six Protocol instances |
| `evaluate_environment` | function | Orbit, plume, look, project, appearance |

## Inputs and outputs

**`evaluate_environment(models, camera, eph, time, shutter, rng, prior_plume) -> EnvSample`**

- Inputs: model bundle, `CameraGeometry`, `EphemerisConfig`, `EnvTime`,
  `ShutterPose`, numpy Generator, optional prior `PlumeState`.
- Output: `EnvSample` with `EnvTruth` and `DriverFeed`.

## Behavior

1. `orbit.state_eci(time.utc_s)` supplies ISS truth.
2. `plume.evaluate(...)` returns `PlumeState`. The pipeline passes `rng` into
   the plume model.
3. Band-plane plumes copy `centroid_band_px` and skip ECEF look and project.
4. ECEF plumes call `look_angles_at` and `optics.project_centroid`.
5. An Earth hit closer than the CoG slant drops the centroid.
6. `appearance.render_feed` writes the driver feed.

## Errors and faults

None after a successful `build_environment`. A geometry miss sets
`centroid_band_px` to `None` and `visible` to false.

## Messages

None.

## Configuration

Reads `EphemerisConfig` epoch, Earth rate, and WGS-84 scalars.

## Constraints

- The pipeline does not read a clock or a gimbal.
- Callers thread `prior_plume` across steps.

## Related documents

- [`sim.environment`](../environment.md)
- [`sim.environment.records`](records.md)
- [`sim.environment.look`](look.md)
- [`sim.environment.models`](models.md)
