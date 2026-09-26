# sim.scene.plume

**Source:** `packages/sim/src/sim/scene/plume.py`
**Kind:** module

## Purpose

The plume scene module renders radiometrically plausible raw mosaic frames. It also builds a
`ScriptedDetector` whose fixed mask yields one stable off-boresight blob each frame.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PLANE_HEIGHT_PX` | constant | Native along-track size (1544 px) |
| `PLANE_WIDTH_PX` | constant | Native lateral size (2064 px) |
| `DETECTOR_HEIGHT_PX` | constant | Upsampled inference height (3088 px) |
| `DETECTOR_WIDTH_PX` | constant | Upsampled inference width (4128 px) |
| `build_frames` | function | Render N uint16 mosaic frames with monotonic `frame_id` |
| `plume_detector` | function | Return a `ScriptedDetector` with a 50x50 unit mask |

## Inputs and outputs

**`build_frames(num_frames, seed=0) -> list[MosaicFrame]`**

- Inputs: frame count, NumPy random seed.
- Output: list of `(3, 1544, 2064)` uint16 RGB planes with exposure and gain metadata.

**`plume_detector() -> ScriptedDetector`**

- Output: detector with mask region `[55:105, 2039:2089]` at the upsampled tensor
  size, confidence gate 0.55, minimum blob area 15 px.

## Behavior

1. `build_frames` builds a Gaussian plume on the native 1544x2064 grid at (x=1032, y=40)
   with sigma 40 px, above boresight.
2. It composites background and per-channel plume amplitudes, adds Gaussian read noise
   (sigma 2 DN), and quantizes to 12-bit.
3. It stores the three planes in `BAND_ORDER`.
4. It assigns `frame_id` values 1 through `num_frames` with fixed timestamp metadata.
5. `plume_detector` fills the upsampled mask with a square at unit probability above
   boresight.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

- The red plane is brightest inside the plume region.
- The plume sits above the native-frame boresight.
- TRACKING commands positive elevation inside the science window.

## Related documents

- [`sim.scene`](scene.md)
- [`sim.sil`](sil.md)
