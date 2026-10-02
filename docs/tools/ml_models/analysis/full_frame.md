# tools.ml_models.analysis.full_frame

**Source:** `packages/tools/src/tools/ml_models/analysis/full_frame.py`
**Kind:** module

## Purpose

This module evaluates a conditioned classifier/segmentor pair on real
finished flight frames. Torch models are adapted to flight's typed NumPy tile
backend protocols, then scored by `flight.payload.inference.detector.infer_tiles`.
Flight also owns GSD encoding and row-major tile slicing and stitching. Frame
dimensions, band count, grid dimensions, and reference GSD come from the
flight-backed dataset geometry aliases.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TiledScore` | dataclass | Stitched probability mask plus row-major logits and positive flags |
| `score_tiled_frame` | function | Classify one unit frame and segment selected tiles |
| `evaluate_flight_frames` | function | Score complete 64-tile test frames of a finished flight dataset |

## Inputs and outputs

`score_tiled_frame(classifier, segmentor, frame, tile_gsd_m, *,
gsd_reference_m, logit_threshold)` takes a float32
`(len(INPUT_BANDS), H, W)` unit frame and `(grid_rows * grid_cols, 2)` metre
GSD pairs, using the flight-backed geometry defaults, and returns `TiledScore`.
`evaluate_flight_frames(dataset, classifier, segmentor, *,
logit_threshold)` returns a JSON-ready dict with per-frame reports,
per-bin aggregates, and `incomplete_frame_ids`.

## Behavior

1. `evaluate_flight_frames` loads `dataset.json`, requires
   `source == "flight"`, and groups test rows by `frame_id` and
   `grid_rc`.
2. Frames missing any configured grid cell are skipped and reported in
   `incomplete_frame_ids`.
3. `score_tiled_frame` uses flight's tiling and GSD helpers, classifies every
   configured grid tile through `infer_tiles`, and lets the shared positive
   gate run the segmentor only on tiles whose logit reaches the threshold.
4. Tiles with a stored ground mask contribute per-tile IoU and Dice; a
   frame with all configured masks also scores stitched `full_frame_iou` and
   `full_frame_dice`. Frames without masks contribute classifier metrics
   only.
5. Per-bin aggregates use the stored `bin_id` of each tile.

## Errors and faults

`ValueError` on a non-flight manifest, non-flight tile dimensions,
missing frame coordinates, duplicate tile coordinates, augmented test
rows, a non-finite threshold, or non-finite model outputs.

## Messages

None.

## Configuration

`logit_threshold` defaults to 0.0; `gsd_reference_m` defaults to the
manifest reference.

## Constraints

- Only complete frames with every configured grid cell are scored.
- No synthetic scene composition or resizing; stored tiles only.
- Unannotated masks remain unknown and are never treated as empty.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.cli`](../cli.md)
- [`tools.ml_models.dataset.geometry`](../dataset/geometry.md)
- [`tools.ml_models.train.metrics`](../train/metrics.md)
