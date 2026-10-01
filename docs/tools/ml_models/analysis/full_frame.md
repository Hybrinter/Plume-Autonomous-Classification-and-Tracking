# tools.ml_models.analysis.full_frame

**Source:** `packages/tools/src/tools/ml_models/analysis/full_frame.py`
**Kind:** module

## Purpose

This module slices a camera frame into an 8 by 8 tile grid, runs that batch
once, and stitches the segmentor probability mask. It scores the stitched
frame with the classifier gate and blob overlap. It also builds plume scenes
and empty scenes from a pack test split.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `tile_hw_for_frame` | function | Tile size for an 8 by 8 grid |
| `slice_tiles` | function | Row-major tiles, along-track then lateral |
| `stitch_tiles` | function | Tile probabilities back to the frame |
| `TiledFrameScore` | class | Gate result, max tile logit, stitched plane |
| `score_tiled_frame` | function | One tile batch, stitch, gate, and blobs |
| `FullFrameScore` | class | Hit, empty-frame false positive, placement |
| `score_full_frame` | function | Gate plus `extract_blobs` overlap |
| `placement_of` | function | Center, corner, edge, or empty |
| `EvalScene` | class | One camera frame plus the source chip |
| `build_eval_scenes` | function | Test-split plume scenes and empty scenes |
| `FrameEval` | class | Score plus chip scores and tile scores |
| `score_dry_run` | function | Fixed tile logits for one scene |
| `summarize_frames` | function | Hit rate, empty-frame rate, chip scores, and tile scores |

## Inputs and outputs

`tile_hw_for_frame(frame_hw) -> tuple[int, int]`. The camera frame
`(1544, 2064)` returns `(193, 258)`.

`slice_tiles(frame, tile_hw=None) -> ndarray`. A camera frame returns 64
tiles of shape `(64, C, 193, 258)` or `(64, 193, 258)`.

`stitch_tiles(tiles, frame_hw) -> ndarray`. The stitched camera mask is
`(1544, 2064)`.

`score_tiled_frame(image, gt_mask, forward, *, logit_threshold=0.0, prob_threshold=0.55, min_area=15, placement=None) -> TiledFrameScore`.

`score_full_frame(logits_classifier, prob_mask, gt_mask, *, logit_threshold=0.0, prob_threshold=0.55, min_area=15, placement=None) -> FullFrameScore`.

`build_eval_scenes(pack_dir, *, limit=1, seed=0, frame_h=None, frame_w=None) -> tuple[EvalScene, ...]`.
The result is `limit` plume scenes, then `limit` empty scenes. The default
frame is 1544 by 2064.

`summarize_frames(records) -> dict`. Keys include `full_frame_hit_rate`,
`hit_rate_by_placement`, `empty_frame_false_positive_rate`, `chip_scores`,
`tile_scores`, and `chip_iou`.

## Behavior

1. `slice_tiles` cuts the frame into an 8 by 8 grid, row-major. The row
   axis is along-track. The column axis is lateral.
2. On a 1544 by 2064 frame each tile is 193 by 258. The batch holds 64 tiles.
3. `score_tiled_frame` calls the tile forward once on that batch.
4. `stitch_tiles` writes each probability tile back to its frame window.
5. The gate is open when the max tile classifier logit is at least 0.
6. `extract_blobs` reads the stitched probability mask. The default
   probability threshold is 0.55. The default minimum area is 15 pixels.
7. A hit requires an open gate and a blob that overlaps the ground truth.
8. An empty ground truth with an open gate and a blob is an empty-frame
   false positive.
9. A max tile logit below 0 is a miss when the mask matches the ground truth.
10. `build_eval_scenes` loads the test split. Chips in that split are square
    and share one side.
11. The function returns `limit` plume scenes, then `limit` empty scenes.
    The frame is `frame_h` by `frame_w`. The default frame is 1544 by 2064.
12. A plume scene places one annotated test chip on a fill of negative test
    chips. The image border blends. The mask keeps the polygon pixels that
    land in the frame. `chip_image` is the first annotated test chip.
13. An empty scene is that fill with an empty mask.
14. `summarize_frames` reports chip scores and tile scores as separate means.
15. `chip_iou` is a side value. It is not a pass or fail field.

## Errors and faults

`ValueError` when a mask is not a plane, the logit array is empty, the frame
does not divide into the 8 by 8 grid, the probability batch length disagrees
with the tile batch, the test split is empty, chips are not square, the test
split has no negative chip, the test split has no annotated chip, or `limit`
is below 1.

## Messages

None.

## Configuration

`prob_threshold` defaults to `PactConfig.controller.vision.confidence_gate`.
`min_area` defaults to `min_blob_area_px`. The camera frame is 1544 by 2064.
The tile grid is 8 by 8. Each flight tile is 193 by 258.

## Constraints

This module imports `flight.payload.blobs`, `flight.libs.config`, and
`tools.ml_models.data.prism`. It does not import `flight.payload.inference`,
`flight.core`, or `tools.analysis`.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.data.prism`](../data/prism.md)
- [`flight.payload.blobs`](../../../flight/payload/blobs.md)
