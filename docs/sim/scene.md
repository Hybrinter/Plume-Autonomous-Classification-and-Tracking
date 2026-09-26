# sim.scene

**Source:** `packages/sim/src/sim/scene/`
**Kind:** package

## Purpose

The scene package renders synthetic imagery for SIL. It produces raw RGB frames and a
scripted detector that yield stable plume blobs for closed-loop tests.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`plume`](scene/plume.md) | module | Gaussian plume RGB frames and `plume_detector` |

## Package interface

`sim.scene.__init__` re-exports:

| Name | Kind |
| --- | --- |
| `build_frames` | function |
| `plume_detector` | function |

## Interactions

Scene code imports `flight.libs.types` and `flight.payload.inference`. It does not
use the message bus.

GSE and tools analysis call `build_frames` and `plume_detector` when they wire a SIL run.

## Constraints

- Frames are deterministic for a given seed.
- The plume sits above boresight on the native RGB grid.
- `ScriptedDetector` reads a fixed probability mask. It does not inspect tensor content.

## Related documents

- [`sim`](sim.md)
- [`sim.scene.plume`](scene/plume.md)
- [`sim.sil`](sil.md)
