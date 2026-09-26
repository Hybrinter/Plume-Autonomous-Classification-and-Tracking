# tools.ml_models.data.canvas

**Source:** `packages/tools/src/tools/ml_models/data/canvas.py`
**Kind:** module

## Purpose

This module builds a flight frame from background chips and an optional
annotated plume, then returns the full frame or a window.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `CanvasConfig` | class | Frame size, window, and paste settings |
| `Chip` | class | One source tile |
| `CanvasSample` | class | Full frame or window |
| `build_scene` | function | Mosaic plus an optional plume |
| `take_window` | function | Crop that keeps a positive pixel in frame |
| `sample_view` | function | Scene, then full frame or window |

## Inputs and outputs

`CanvasConfig` defaults: `frame_hw=(1544, 2064)`, `window_px=512`,
`full_frame_every=8`, `chip_side=76`, `empty_fraction=0.5`, `max_plumes=1`,
`feather_px=6`, `seed=0`.

`Chip` holds image `(C, H, W)` float32, mask `(1, H, W)` float32 or empty,
`label`, `group_id`, `split`, and `annotated`.

`build_scene(chips, config, rng) -> tuple[np.ndarray, np.ndarray, float]`.
The image is `(C, frame_h, frame_w)`. The mask is `(1, frame_h, frame_w)`.

`take_window(scene_image, scene_mask, config, rng) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]`.
The third value is `(row, column)`.

`sample_view(chips, config, rng, *, full_frame) -> CanvasSample`.
`origin` is `None` when `full_frame` is True.

## Behavior

1. Background chips have `label == 0` and one shared split. The mosaic uses a
   random phase. Each tile copies one background chip. The chip is not scaled.
2. A positive chip whose split differs from the background split raises.
3. With probability `1 - empty_fraction`, and when `max_plumes` is at least 1,
   the scene pastes that many annotated chips. The offset is uniform on
   `[-chip_h + 1, frame_h)` and `[-chip_w + 1, frame_w)`.
4. The image blend uses `feather_paste`. The mask receives the polygon values
   that land in the frame. The mask is not feathered and is not the chip
   rectangle unless the polygon fills the chip.
5. An unannotated chip raises when the scene tries to paste it.
6. The scene label is 1 when any mask pixel is 1. Otherwise the label is 0.
7. An empty window is a uniform crop inside the frame. A positive window
   picks one positive pixel and places it at a uniform coordinate inside the
   window, including the border. The origin is clamped. The window stays
   inside the frame.
8. When `window_px` is greater than or equal to the frame height or the frame
   width, `take_window` returns the full frame at origin `(0, 0)`.
9. `sample_view` builds one scene. `full_frame` True returns that scene.
   `full_frame` False returns `take_window`. `config.seed` and
   `full_frame_every` are not read here.

## Errors and faults

`ValueError` when the config is out of range, chips disagree on shape, the
background split is mixed, a positive split differs, or an unannotated chip
is pasted.

## Messages

None.

## Configuration

`CanvasConfig` fields are constructor arguments. There is no TOML file.
`empty_fraction` is in `[0, 1]`. `chip_side` matches each chip's height and
width.

## Constraints

Callers pass the Generator. A scene uses one split. Unannotated positives are
not pasted. The default frame is 1544 by 2064. Callers may pass a smaller
`frame_hw`.

## Related documents

- [`tools.ml_models.data`](../data.md)
- [`tools.ml_models.data.augment`](augment.md)
- [`tools.ml_models.data.prism`](prism.md)
