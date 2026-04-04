"""
Blob tracker for PACT controller subsystem.

Associates blobs across consecutive inference frames using Intersection-over-Union (IoU)
matching. Persistent blob IDs allow the arbiter and EMA filter to track individual targets
across frames.

Satisfies: REQ-AIML-DATA-006
"""

from __future__ import annotations

from pact.types.messages import BlobMeta


def compute_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Compute Intersection-over-Union between two axis-aligned bounding boxes.

    Both boxes are in (x_min, y_min, x_max, y_max) pixel-space format.
    Returns a value in [0.0, 1.0]. Returns 0.0 for zero-area boxes.

    Parameters
    ----------
    box_a, box_b:
        Bounding boxes as (x_min, y_min, x_max, y_max). Coordinates are inclusive
        pixel indices in the cropped tensor frame.

    Returns
    -------
    float
        IoU score in [0.0, 1.0].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection rectangle
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    # Width/height of intersection (0 if no overlap)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    if intersection == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def match_blobs(
    prev_blobs: tuple[BlobMeta, ...],
    new_blobs: tuple[BlobMeta, ...],
    iou_threshold: float,
) -> tuple[BlobMeta, ...]:
    """Associate blobs across frames by IoU. REQ-AIML-DATA-006.

    Matching algorithm
    ------------------
    For each new blob, find the previous blob with the highest IoU score that meets
    or exceeds `iou_threshold`. If a match is found:
      - The new blob inherits the `blob_id` from the matched previous blob.
      - `persistence_count` is incremented by 1 (carried from the previous blob).
    If no match is found:
      - The new blob is assigned a fresh `blob_id` (max existing ID + 1, or 1 if none).
      - `persistence_count` is set to 1.

    Each previous blob can be matched to at most one new blob (greedy, first-come
    first-served on the sorted new_blobs list). Unmatched previous blobs are dropped.

    Parameters
    ----------
    prev_blobs:
        Blobs from the previous frame (may be empty on first frame).
    new_blobs:
        Blobs from the current inference result, before ID assignment.
    iou_threshold:
        Minimum IoU for a match to be accepted (from ControllerConfig.blob_iou_match_threshold).

    Returns
    -------
    tuple[BlobMeta, ...]
        New blobs with updated blob_ids and persistence_counts.
    """
    ...
