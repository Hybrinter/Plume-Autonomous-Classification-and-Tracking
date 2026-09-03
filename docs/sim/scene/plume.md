# sim.scene.plume

**Source:** `packages/sim/src/sim/scene/plume.py`
**Kind:** module

## Purpose

The plume scene module renders radiometrically plausible raw mosaic frames. It also builds a
`ScriptedDetector` whose fixed mask yields one stable off-boresight blob each frame.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `MOSAIC_HEIGHT_PX` | constant | Mosaic along-track size (2048 px) |
| `MOSAIC_WIDTH_PX` | constant | Mosaic lateral size (2448 px) |
| `BAND_HEIGHT_PX` | constant | Band-plane along-track size (1024 px) |
| `BAND_WIDTH_PX` | constant | Band-plane lateral size (1224 px) |
| `DETECTOR_HEIGHT_PX` | constant | Inference tensor height (1024 px) |
| `DETECTOR_WIDTH_PX` | constant | Inference tensor width (1224 px) |
| `build_frames` | function | Render N uint16 mosaic frames with monotonic `frame_id` |
| `plume_detector` | function | Return a `ScriptedDetector` with a 50x50 unit mask |

## Inputs and outputs

**`build_frames(num_frames, seed=0) -> list[MosaicFrame]`**

- Inputs: frame count, NumPy random seed.
- Output: list of `(2048, 2448)` uint16 mosaic planes with exposure and gain metadata.

**`plume_detector() -> ScriptedDetector`**

- Output: detector with mask region `[875:925, 587:637]` at tensor resolution, confidence
  gate 0.55, minimum blob area 15 px.

## Behavior

1. `build_frames` builds a 1024x1224 Gaussian plume in band-plane space at (x=612, y=900)
   with sigma 40 px.
2. It composites background and per-band plume amplitudes, adds Gaussian read noise (sigma
   2 DN), and quantizes to 12-bit.
3. It interleaves four band planes into the 2x2 CFA mosaic via `interleave_bands`.
4. It assigns `frame_id` values 1 through `num_frames` with fixed timestamp metadata.
5. `plume_detector` fills a 1024x1224 float mask with a square at unit probability below
   boresight.

## Errors and faults

None. Fixed geometry makes `interleave_bands` succeed with an internal assert.

## Messages

None.

## Configuration

None.

## Constraints

- NIR band amplitude is highest inside the plume region.
- The centroid sits ~388 px below the 1024x1224-plane boresight (612, 512).
- TRACKING commands negative elevation. Drivers pin azimuth at 0.

## Related documents

- [`sim.scene`](scene.md)
- [`sim.sil`](sil.md)
