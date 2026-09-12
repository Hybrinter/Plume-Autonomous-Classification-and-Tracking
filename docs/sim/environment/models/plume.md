# sim.environment.models.plume

**Source:** `packages/sim/src/sim/environment/models/plume.py`
**Kind:** module

## Purpose

The plume module supplies the `PlumeModel` Protocol and named band-plane and
ECEF-column implementations.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PlumeModel` | Protocol | `evaluate(...) -> PlumeState` |
| `BandplaneGaussian` | class | Fixed CI band-plane centroid |
| `EcefColumn` | class | Frozen ECEF CoG at the height proxy |

## Inputs and outputs

**`PlumeModel.evaluate(time, prior, wind, earth, iss, orbit, omega, epoch) -> PlumeState`**

- Inputs: sample time, optional prior state, wind, Earth, ISS, orbit, Earth
  rate, epoch UTC.
- Output: `PlumeState` in `ecef` or `bandplane`.

## Behavior

1. `BandplaneGaussian` returns centroid `(612, 124)` px and no ECEF CoG.
2. `EcefColumn` with a stored CoG returns that CoG.
3. `EcefColumn` with `cog_ecef_m is None` places the CoG at the nadir
   height-proxy hit.

## Errors and faults

None. A nadir miss sets `cog_ecef_m` to `None`.

## Messages

None.

## Configuration

`EnvironmentConfig.ecef_column` supplies CoG, sigmas, and height proxy.

## Constraints

`BandplaneGaussian` ignores Earth, orbit, wind, and shutter elevation.

## Related documents

- [`sim.environment.models`](../models.md)
- [`sim.environment.evaluate`](../evaluate.md)
- [`sim.scene.plume`](../../scene/plume.md)
