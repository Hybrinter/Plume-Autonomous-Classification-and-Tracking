# flight.payload.tracking.tracker

**Source:** `packages/flight/src/flight/payload/tracking/tracker.py`
**Kind:** pure module

## Purpose

The tracker associates blobs across frames with IoU matching. It assigns stable blob IDs and
increments persistence counts.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `compute_iou` | function | IoU between two axis-aligned boxes |
| `match_blobs` | function | Greedy cross-frame association |

## Inputs and outputs

`compute_iou(box_a, box_b)` returns a float in `[0, 1]`. Boxes are `(x_min, y_min, x_max, y_max)`.

`match_blobs(prev_blobs, new_blobs, iou_threshold)` returns an updated blob tuple.

## Behavior

1. Build candidate pairs with IoU at or above the threshold.
2. Sort candidates by IoU descending and match greedily; each prev and new blob matches at most
   once.
3. On match, copy `blob_id` and set `persistence_count = prev + 1`.
4. On no match, assign a new id as max existing id plus one and set persistence to 1.
5. Empty `new_blobs` returns an empty tuple.

## Errors and faults

None.

## Messages

None.

## Configuration

Reads `ControllerConfig.blob_iou_match_threshold`.

## Constraints

Pure module. Initial detection sets `blob_id` and `persistence_count` to zero in
`extract_blobs`; this module assigns real values before the arbiter runs.

## Related documents

- [`flight.payload.tracking`](tracking.md)
- [`flight.payload.model.blobs`](model/blobs.md)
- [`flight.payload.gimbal.arbiter`](gimbal/arbiter.md)
