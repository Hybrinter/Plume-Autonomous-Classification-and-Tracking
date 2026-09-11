# ADR-REPO-0014: World twin as a second selection space beside HAL axes

**Status:** Proposed
**Date:** 2026-09-11
**Topic:** sim-fidelity
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-REPO-0010, ADR-REPO-0008

## Context

`EnvironmentConfig` plus `select_drivers` already selects payload devices per axis
(`sensor`, `gimbal`, `ephemeris`, `compute`, `link`, `clock`) as `sim` or `real`.
That matrix answers which driver talks to a flight app.

Performance studies for the selection committee need a different question: which
world those drivers look at. Earth shape, truth orbit, wind, plume geometry,
pinhole projection, and silhouette formation are not payload devices. They change
closed-loop error, smear, and detect rate. Today they are static (`sim.scene.plume`
paints a fixed band-plane Gaussian), empty (`sim.twin/`), or duplicated in
`analysis.lib` in kilometres, which SIL cannot import as a shared substrate.

Folding those objects into `EnvironmentConfig` would put world models on the flight
image and would overload `sim`/`real` until it meant both "device stand-in" and
"physics fidelity." Putting them only in analysis studies would keep two geometries
(flight `geo`/`intersect` vs analysis orbit/look/optics) and would not close the
gimbal-to-scene loop.

## Decision

Keep two orthogonal selection spaces:

1. **HAL axes** stay in `EnvironmentConfig`: each payload device is `sim` or `real`.
   `select_drivers` is unchanged in role. SIL/PIL/HIL remain named corners of this
   matrix.
2. **World axes** live in a sim-only `TwinConfig`: each non-device object is a
   **named** physics model (`circular_kepler`, `wgs84_ellipsoid`, `pinhole`, …),
   not a `low`/`med`/`high` rung and not a `sim`/`real` bit.

The twin emits truth records. The SIL composition root pushes a driver feed
(mosaic, optional mask) into sim HAL drivers. Flight apps see only HAL
observations. HAL ephemeris stays its own driver; twin ISS state is never the
payload predictor input. Analysis scores flight telemetry against twin truth.

`sim.twin` composes the models. Binding lives in `sim.sil` (`SilTwinBind.pre_step`
on both SIL harnesses): push a mosaic via `SimSensor.load_next` and, when compute
is scripted, a mask via `ScriptedDetector.load_mask`. Do not add a `FrameSource`
HAL Protocol. Flight never imports `sim`. World config never enters `PactConfig`.

Monte Carlo, clocks, and trial loops sit around the twin in analysis/tools. They
are not world models.

The conceptual specification is `docs/design/scene-twin.md`. Implementation follows
that brief in a later change. This record is Proposed until the first `sim.twin`
slice lands and STE pages match.

## Consequences

- Selection-committee studies can mix HAL corners with named world models without
  forking the flight apps.
- `analysis.lib` geometry is no longer the closed-loop authority; new studies
  consume `sim.twin`. Existing analysis studies may keep their helpers until a
  migration.
- A small `SimSensor.load_next` mutator (or equivalent) is required so the SIL
  root can push live frames without `flight` importing `sim`. Scripted compute
  needs a matching `load_mask`. The bind must run on `ValidationHarness` as well
  as `SilHarness`, or GSE studies never sample the twin.
- CI default scenes stay pre-rendered `static_gaussian_bandplane` until a study
  opts into closed-loop sampling.
- Operators of the flight image never see TwinConfig.

## Alternatives considered

- **Extend `EnvironmentConfig` with world fields.** Rejected: flight image has no
  Earth model; `sim`/`real` cannot name fidelity.
- **One "fidelity" enum on the SIL harness.** Rejected: mix-and-match per axis is
  the point; a single rung hides which physics moved the metric.
- **Put world sampling inside each sim HAL driver.** Rejected: import direction
  (`flight` cannot import `sim`) and would couple device stand-ins to a scene.
- **Keep physics only in `packages/analysis`.** Rejected: SIL and GSE could not
  share the substrate; units and epoch conventions would keep drifting.
