# flight.payload.model.blobs

**Source:** `packages/flight/src/flight/payload/model/blobs.py`
**Kind:** pure module

## Purpose

This module extracts connected-component blobs from a segmentation probability mask.
Both detector backends call it so blob geometry stays identical across scripted and
ONNX paths.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `extract_blobs` | function | Labels components and builds `BlobMeta` records |

## Inputs and outputs

`extract_blobs(prob_mask, confidence_gate, min_blob_area_px)` takes a `(H, W)` float32
probability map. It returns `tuple[BlobMeta, ...]` with `blob_id` and
`persistence_count` set to zero.

## Behavior

1. Threshold the mask at `confidence_gate` to form a binary image.
2. Label connected components with `scipy.ndimage.label`.
3. For each component, skip when pixel area is below `min_blob_area_px`.
4. Compute axis-aligned bbox, centroid, mean confidence, and pixel area.
5. Append a `BlobMeta` for each surviving component.

## Errors and faults

None.

## Messages

None.

## Configuration

Thresholds come from `ControllerConfig.confidence_gate` and `min_blob_area_px`, passed
by the detector at construction time.

## Constraints

The function is pure. Blob IDs and persistence counts are assigned later by
`match_blobs` in the tracking package.

## Related documents

- [`flight.payload.model.detector`](detector.md)
- [`flight.payload.tracking.tracker`](../tracking/tracker.md)
