# flight.payload.model.blobs

**Source:** `packages/flight/src/flight/payload/model/blobs.py`
**Kind:** pure module

## Purpose

This module extracts connected-component blobs from a segmentation probability mask. Both detector
backends share this geometry.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `extract_blobs` | function | Labels components and builds `BlobMeta` records |

## Inputs and outputs

`extract_blobs(prob_mask, confidence_gate, min_blob_area_px)` takes a `(H, W)` float32 mask in
`[0, 1]`. It returns a tuple of `BlobMeta` with `blob_id` and `persistence_count` set to 0.

## Behavior

1. Threshold the mask at `confidence_gate` to a binary image.
2. Label connected components with `scipy.ndimage.label`.
3. For each component, skip areas below `min_blob_area_px`.
4. Compute bounding box, centroid, pixel area, and mean confidence inside the component.
5. Append a `BlobMeta` for each surviving component.

## Errors and faults

None.

## Messages

None.

## Configuration

Thresholds come from the detector constructor or `ControllerConfig` fields passed at build time.

## Constraints

Pure module. Persistence and blob IDs are assigned later by `match_blobs`.

## Related documents

- [`flight.payload.model`](model.md)
- [`flight.payload.model.detector`](detector.md)
- [`flight.payload.tracking.tracker`](tracking/tracker.md)
