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
   A new draw uses an exponential along-track gap at the intensity at the
   current sub-satellite latitude,
   `dens(lat) * 2 * cross_track_half_km`. Placement walks the inertial LVLH
   `+x` arc at the draw instant, then hits the Earth along the geocentric
   radial at the height proxy. A gap of π rad or more of that arc is absent.
   Zero density sets `present` false.

## Errors and faults

None. A nadir miss or a placement failure sets `cog_ecef_m` to `None` and
`present` false.

## Messages

None.

## Configuration

`EnvironmentConfig.ecef_column` supplies CoG, sigmas, and height proxy.
`EnvironmentConfig.poisson_latitude` supplies signed-latitude density tables
and corridor width.

## Constraints

`BandplaneGaussian` ignores Earth, orbit, wind, rng, and shutter elevation.
`PoissonLatitude` does not model rewind hunt or azimuth raster. Placement
does not apply Earth rotation to the inertial arc.

## Related documents

- [`sim.environment.models`](../models.md)
- [`sim.environment.evaluate`](../evaluate.md)
- [`sim.scene.plume`](../../scene/plume.md)
