# sim

**Source:** `packages/sim/src/sim/`
**Kind:** package

## Purpose

The sim package supplies synthetic scenes and the software-in-the-loop (SIL) harness. It runs
the real flight apps over sim HAL drivers and steps them deterministically.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`scene`](sim/scene.md) | package | Synthetic mosaic frames and scripted detections |
| [`sil`](sim/sil.md) | package | SIL and validation harness builders and steppers |
| [`twin`](sim/twin.md) | stub | Deferred dynamics twin scaffold (empty) |

## Package interface

`sim` has no top-level `__init__.py` exports. Import from `sim.scene`, `sim.sil`, or
`sim.sil.stepping`.

The SIL harness shares one cycle body: `sim.sil.stepping.step_once`. GSE and analysis reuse
that function.

Driver selection uses `EnvironmentConfig` from `flight.libs.config`. Command-path tests sign
packets with `flight.libs.commands.build_tc_packet`.

## Interactions

The package imports flight core, HAL protocols, payload, and fault modules. It constructs
sim drivers and calls `flight.core.composition.build_apps` with the same wiring path as
flight.

GSE imports `sim.sil` for `build_validation_system`, `ValidationHarness`, and `step_once`.
Tools analysis imports `sim.scene` and `sim.sil` for passive capture runs.

## Constraints

- SIL runs the real flight apps. There is no parallel app graph.
- `build_sil_system` forces every environment axis to `"sim"`.
- `build_validation_system` passes `config.environment` through unchanged.
- The harness replaces the thread scheduler. It publishes heartbeats manually each step.
- `sim/twin/` is an empty stub. No scene-feedback dynamics twin exists yet.

## Related documents

- [`sim.scene`](sim/scene.md)
- [`sim.sil`](sim/sil.md)
- [`sim.twin`](sim/twin.md)
- [`gse`](gse.md)
- [`tools`](tools.md)
