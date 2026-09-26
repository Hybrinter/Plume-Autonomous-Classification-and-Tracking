# tools.ml_models.analysis.full_frame

**Source:** `packages/tools/src/tools/ml_models/analysis/full_frame.py`
**Kind:** module

## Purpose

This module scores one full frame with the classifier gate and blob overlap,
and it builds canvas scenes from a pack test split.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `FullFrameScore` | class | Hit, empty-frame false positive, placement |
| `score_full_frame` | function | Gate plus `extract_blobs` overlap |
| `placement_of` | function | Center, corner, edge, or empty |
| `EvalScene` | class | One full-frame canvas view |
| `build_eval_scenes` | function | Test-split plume scenes and empty scenes, or draws from a canvas |
| `default_canvas` | function | Canvas with a 1544 by 2064 default frame |
| `FrameEval` | class | Score plus chip and frame logits |
| `score_dry_run` | function | Fixed logits for one scene |
| `summarize_frames` | function | Hit rate, empty-frame rate, and margin |

## Inputs and outputs

`score_full_frame(logits_classifier, prob_mask, gt_mask, *, logit_threshold=0.0, prob_threshold=0.55, min_area=15, placement=None) -> FullFrameScore`.

`build_eval_scenes(pack_dir, canvas=None, *, limit=1, seed=0, frame_h=None, frame_w=None) -> tuple[EvalScene, ...]`.
An omitted canvas yields `limit` plume scenes and `limit` empty scenes.
An explicit canvas yields `limit` draws from that canvas.

`summarize_frames(records) -> dict`. Keys include `full_frame_hit_rate`,
`hit_rate_by_placement`, `empty_frame_false_positive_rate`,
`chip_vs_frame_logit_margin`, and `chip_iou`.

## Behavior

1. The gate is open when the max classifier logit is at least 0.
2. `extract_blobs` reads the probability mask. The default probability
   threshold and minimum area are the controller vision gates.
3. A hit requires an open gate and a blob that overlaps the ground truth.
4. An empty ground truth with an open gate and a blob is an empty-frame
   false positive.
5. A logit below 0 is a miss even when the mask matches the ground truth.
6. `build_eval_scenes` loads the test split and calls `sample_view` with
   `full_frame` true.
7. An omitted canvas yields `limit` plume scenes and `limit` empty scenes.
8. An explicit canvas yields `limit` draws from that canvas.
9. `summarize_frames` reports `chip_iou` as a side value. It is not a pass
   or fail field.

## Errors and faults

`ValueError` when a mask is not a plane, the logit array is empty, the test
split is empty, or `limit` is below 1.

## Messages

None.

## Configuration

`prob_threshold` defaults to `PactConfig.controller.vision.confidence_gate`.
`min_area` defaults to `min_blob_area_px`. The default frame in
`default_canvas` is 1544 by 2064.

## Constraints

This module imports `flight.payload.blobs` and `flight.libs.config`. It does
not import `flight.payload.inference`, `flight.core`, or `tools.analysis`.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.data.canvas`](../data/canvas.md)
- [`flight.payload.blobs`](../../../flight/payload/blobs.md)
