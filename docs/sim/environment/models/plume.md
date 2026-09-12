# sim.environment.models.plume

**Source:** `packages/sim/src/sim/environment/models/plume.py`
**Kind:** module

## Purpose

The plume module supplies the `PlumeModel` Protocol and named band-plane,
ECEF-column, and latitude Poisson implementations.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PlumeModel` | Protocol | `evaluate(...) -> PlumeState` |
| `BandplaneGaussian` | class | Fixed CI band-plane centroid |
| `EcefColumn` | class | Frozen ECEF CoG at the height proxy |
| `PoissonLatitude` | class | Along-track Poisson CoG at signed-latitude density |
| `along_track_intensity_per_km` | function | 1-D intensity dens * corridor width |

## Inputs and outputs

**`PlumeModel.evaluate(time, prior, wind, earth, iss, orbit, omega, epoch, rng) -> PlumeState`**

- Inputs: sample time, optional prior state, wind, Earth, ISS, orbit, Earth
  rate, epoch UTC, numpy Generator.
- Output: `PlumeState` in `ecef` or `bandplane`.

## Behavior

1. `BandplaneGaussian` returns centroid `(612, 124)` px, no ECEF CoG, and
   `present` true.
2. `EcefColumn` with a stored CoG returns that CoG and `present` true.
3. `EcefColumn` with `cog_ecef_m is None` places the CoG at the nadir
   height-proxy hit.
4. `PoissonLatitude` keeps the prior CoG while it stays ahead of the ISS.
   A new draw uses an exponential along-track gap at intensity
   `dens(lat) * 2 * cross_track_half_km`. Zero density sets `present` false.

## Errors and faults

None. A nadir miss sets `cog_ecef_m` to `None`.

## Messages

None.

## Configuration

`EnvironmentConfig.ecef_column` supplies CoG, sigmas, and height proxy.
`EnvironmentConfig.poisson_latitude` supplies signed-latitude density tables
and corridor width.

## Constraints

`BandplaneGaussian` ignores Earth, orbit, wind, rng, and shutter elevation.
`PoissonLatitude` does not model rewind hunt or azimuth raster.

## Related documents

- [`sim.environment.models`](../models.md)
- [`sim.environment.evaluate`](../evaluate.md)
- [`sim.scene.plume`](../../scene/plume.md)
