# sim.scene.plume

**Source:** `packages/sim/src/sim/scene/plume.py`
**Kind:** module

## Purpose

The plume scene module renders radiometrically plausible raw mosaic frames. It also builds a
`ScriptedDetector` whose fixed mask yields one stable off-center blob each frame.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `FRAME_SIZE` | constant | Mosaic plane size (1024 px) |
| `DETECTOR_SIZE` | constant | Inference tensor size (512 px) |
| `build_frames` | function | Render N uint16 mosaic frames with monotonic `frame_id` |
| `plume_detector` | function | Return a `ScriptedDetector` with a 50x50 unit mask |

## Inputs and outputs

**`build_frames(num_frames, seed=0) -> list[MosaicFrame]`**

- Inputs: frame count, NumPy random seed.
- Output: list of `(1024, 1024)` uint16 mosaic planes with exposure and gain metadata.

**`plume_detector() -> ScriptedDetector`**

- Output: detector with mask region `[315:365, 315:365]` on the 512 plane, confidence
  gate 0.55, minimum blob area 15 px.

## Behavior

1. `build_frames` builds a 512x512 Gaussian plume in band-plane space at (340, 340) with
   sigma 24 px.
2. It composites background and per-band plume amplitudes, adds Gaussian read noise (sigma
   2 DN), and quantizes to 12-bit.
3. It interleaves four band planes into the 2x2 CFA mosaic via `interleave_bands`.
4. It assigns `frame_id` values 1 through `num_frames` with fixed timestamp metadata.
5. `plume_detector` fills a 512x512 float mask with a square at unit probability around
   the same plane coordinates.

## Errors and faults

None. Fixed geometry makes `interleave_bands` succeed with an internal assert.

## Messages

None.

## Configuration

None.

## Constraints

- NIR band amplitude is highest inside the plume region.
- Identity crop and scale 1 map the mask onto the band plane with no back-projection.
- The centroid sits ~83 px off the 512-plane boresight.

## Related documents

- [`sim.scene`](scene.md)
- [`sim.sil`](sil.md)
