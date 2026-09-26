# sim.scene.plume

**Source:** `packages/sim/src/sim/scene/plume.py`
**Kind:** module

## Purpose

The plume scene module renders three registered planes at the AP-3200T frame size.
It also builds a `ScriptedDetector` whose fixed mask yields one stable off-boresight
blob each frame.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `FRAME_HEIGHT_PX` | constant | Along-track size (1544 px) |
| `FRAME_WIDTH_PX` | constant | Lateral size (2064 px) |
| `build_frames` | function | Render N uint16 `(3, H, W)` frames with monotonic `frame_id` |
| `plume_detector` | function | Return a `ScriptedDetector` with one unit-probability rectangle |

## Inputs and outputs

**`build_frames(num_frames, seed=0) -> list[MosaicFrame]`**

- Inputs: frame count, NumPy random seed.
- Output: list of `(3, 1544, 2064)` uint16 buffers in RED, GREEN, BLUE order, with
  exposure and gain metadata.

**`plume_detector() -> ScriptedDetector`**

- Output: detector whose mask is the full frame. The unit rectangle is the old 50 px
  box scaled onto this frame and centered on the plume. Confidence gate 0.55, minimum
  blob area 15 px.

## Behavior

1. `build_frames` places a Gaussian at the same fraction off boresight as pixel
   (612, 124) on a 1224 x 1024 plane. Boresight on this frame is (1032, 772).
2. It composites background and per-channel plume amplitudes, adds Gaussian read noise
   (sigma 2 DN), and quantizes to 12-bit.
3. It stores the three planes as a channel-major buffer. Wire order is RED, GREEN, BLUE.
4. It assigns `frame_id` values 1 through `num_frames` with fixed timestamp metadata.
5. `plume_detector` fills a 1544 x 2064 float mask with one rectangle at unit probability
   above boresight.

## Errors and faults

None.

## Messages

None.

## Configuration

None. Frame size matches `SensorConfig` width and height.

## Constraints

- The red channel carries the strongest plume amplitude.
- The centroid keeps the old fractional offset from boresight.
- TRACKING commands positive elevation inside the science window.

## Related documents

- [`sim.scene`](../scene.md)
- [`sim.sil`](../sil.md)
