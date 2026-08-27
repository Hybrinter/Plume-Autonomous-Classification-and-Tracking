# flight.payload.tracking.tracker

**Source:** `packages/flight/src/flight/payload/tracking/tracker.py`
**Kind:** pure module

## Purpose

This module associates blobs across consecutive inference frames using intersection over
union matching. Persistent blob IDs and frame counts feed the arbiter acquire and
release logic.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `compute_iou` | function | IoU between two axis-aligned bounding boxes |
| `match_blobs` | function | Greedy IoU matching with ID and persistence assignment |

## Inputs and outputs

`compute_iou(box_a, box_b)` takes `(x_min, y_min, x_max, y_max)` boxes and returns a
float in [0, 1].

`match_blobs(prev_blobs, new_blobs, iou_threshold)` returns `tuple[BlobMeta, ...]` with
updated `blob_id` and `persistence_count` on each blob.

## Behavior

1. Build candidate (IoU, prev_index, new_index) pairs at or above the threshold.
2. Sort candidates by IoU descending and greedily match one-to-one.
3. Matched blobs inherit the previous `blob_id` and increment `persistence_count`.
4. Unmatched new blobs receive a fresh id and `persistence_count=1`.
5. Unmatched previous blobs are dropped.

## Errors and faults

None.

## Messages

None.

## Configuration

Uses `ControllerConfig.blob_iou_match_threshold` (default 0.25).

## Constraints

Matching runs after confidence and area gates in `PayloadController`. Blob IDs start at
zero from `extract_blobs` and are assigned here before the arbiter reads persistence.

## Related documents

- [`flight.payload.tracking`](../tracking.md)
- [`flight.payload.blobs`](../blobs.md)
- [`flight.payload.gimbal.arbiter`](../gimbal/arbiter.md)
