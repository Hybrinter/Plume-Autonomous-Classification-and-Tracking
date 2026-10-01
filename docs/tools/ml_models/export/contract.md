# tools.ml_models.export.contract

**Source:** `packages/tools/src/tools/ml_models/export/contract.py`
**Kind:** module

## Purpose

This module carries the GSD-conditioned model markers and the flight GSD
coverage requirement. The shape-verification formulas live in
`flight.payload.inference.contract` and are re-exported here for callers;
they are not duplicated.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `CONDITIONING_ID` | constant | `film-log-gsd-v1` conditioning marker |
| `GSD_ENCODING` | constant | `ln_metres_over_reference_lateral_along` |
| `GsdCoverage` | dataclass | `minimum_m`/`maximum_m` (lateral, along) GSD bounds plus altitude |
| `required_gsd_coverage` | function | Required GSD span from configured geometry |
| `coverage_ok` | function | Spanning check with a 1 mm absolute tolerance |
| `verify_conditioned_shapes` | re-export | Flight two-input shape verifier |

## Behavior

1. `required_gsd_coverage` builds the nominal ISS state at the given altitude
   (default 460 km), evaluates `tile_gsd_grid` on the configured 8x8 grid at
   the science minimum and maximum elevations, and returns per-axis minimum
   and maximum GSD. It returns `None` when the geometry cannot be computed.
2. `coverage_ok` requires finite, ordered, positive actual bounds that span
   the required interval within tolerance.

## Interactions

Reads `PactConfig` (sensor, gimbal, ephemeris) and flight footprint geometry.
Pure computation: no artifact I/O.

## Errors and faults

`required_gsd_coverage` returns `None` on an invalid orbit or failed grid.
`coverage_ok` returns `False` on malformed or insufficient coverage.

## Inputs and outputs

`required_gsd_coverage(cfg=None, altitude_m=460_000.0)` returns a `GsdCoverage` or `None`.

`coverage_ok(minimum_m, maximum_m, required)` returns `bool`.

## Messages

None.

## Configuration

Reads `PactConfig` sensor, gimbal, and ephemeris fields when `cfg` is `None`.

## Constraints

- Shape formulas are owned by `flight.payload.inference.contract`; this module re-exports the verifier.

## Related documents

- [tools.ml_models.export](../export.md)
- [flight.payload.inference.contract](../../../flight/payload/inference/contract.md)
