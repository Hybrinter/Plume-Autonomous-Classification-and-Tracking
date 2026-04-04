"""Unit tests for pact.controller.tracker — compute_iou() and match_blobs().

Satisfies: §6.2 of PACT_SW_ARCH.md — Controller subsystem unit tests.
REQ-AIML-DATA-006
"""

from __future__ import annotations

# third-party
import pytest

# module under test
from pact.controller.tracker import compute_iou, match_blobs

# pact types
from pact.types.messages import BlobMeta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_blob(
    blob_id: int = 1,
    bbox: tuple[int, int, int, int] = (0, 0, 10, 10),
    mean_confidence: float = 0.85,
    pixel_area: int = 100,
    persistence_count: int = 1,
    centroid_raw: tuple[float, float] = (5.0, 5.0),
) -> BlobMeta:
    """Construct a BlobMeta for tracker tests."""
    return BlobMeta(
        blob_id=blob_id,
        bbox=bbox,
        centroid_raw=centroid_raw,
        pixel_area=pixel_area,
        mean_confidence=mean_confidence,
        persistence_count=persistence_count,
    )


# ---------------------------------------------------------------------------
# compute_iou tests
# ---------------------------------------------------------------------------


def test_iou_exact_overlap() -> None:
    """Identical bounding boxes must return IoU = 1.0."""
    box = (10, 10, 50, 50)
    assert compute_iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap() -> None:
    """Completely non-overlapping boxes must return IoU = 0.0."""
    box_a = (0, 0, 10, 10)
    box_b = (20, 20, 30, 30)
    assert compute_iou(box_a, box_b) == pytest.approx(0.0)


def test_iou_partial_overlap() -> None:
    """Partially overlapping boxes must return IoU strictly between 0 and 1."""
    box_a = (0, 0, 10, 10)   # area = 100
    box_b = (5, 5, 15, 15)   # area = 100; overlap = (5,5,10,10) = 25
    score = compute_iou(box_a, box_b)
    assert 0.0 < score < 1.0
    # intersection=25, union=175, IoU ≈ 25/175
    assert score == pytest.approx(25.0 / 175.0, rel=1e-5)


def test_iou_zero_area_box() -> None:
    """A zero-area box (point) must return IoU = 0.0."""
    point = (5, 5, 5, 5)
    box = (0, 0, 10, 10)
    assert compute_iou(point, box) == pytest.approx(0.0)


def test_iou_symmetric() -> None:
    """IoU(a, b) must equal IoU(b, a)."""
    box_a = (0, 0, 20, 20)
    box_b = (10, 10, 30, 30)
    assert compute_iou(box_a, box_b) == pytest.approx(compute_iou(box_b, box_a))


def test_iou_result_in_unit_interval() -> None:
    """IoU result must always be in [0.0, 1.0]."""
    pairs = [
        ((0, 0, 10, 10), (5, 5, 15, 15)),
        ((0, 0, 100, 100), (50, 50, 200, 200)),
        ((0, 0, 1, 1), (0, 0, 1, 1)),
    ]
    for box_a, box_b in pairs:
        score = compute_iou(box_a, box_b)
        assert 0.0 <= score <= 1.0, f"IoU={score} out of [0, 1] for {box_a}, {box_b}"


# ---------------------------------------------------------------------------
# match_blobs tests
# ---------------------------------------------------------------------------


def test_blob_id_persistence() -> None:
    """A new blob that IoU-matches a previous blob inherits the previous blob's ID."""
    prev = (make_blob(blob_id=42, bbox=(0, 0, 10, 10), persistence_count=3),)
    # New blob at same location — should match and inherit blob_id=42
    new = (make_blob(blob_id=99, bbox=(0, 0, 10, 10), persistence_count=1),)
    result = match_blobs(prev, new, iou_threshold=0.5)
    assert len(result) == 1
    assert result[0].blob_id == 42, f"Expected blob_id=42, got {result[0].blob_id}"


def test_blob_persistence_count_incremented() -> None:
    """A matched blob's persistence_count must be previous + 1."""
    prev = (make_blob(blob_id=1, bbox=(0, 0, 10, 10), persistence_count=4),)
    new = (make_blob(blob_id=99, bbox=(0, 0, 10, 10), persistence_count=1),)
    result = match_blobs(prev, new, iou_threshold=0.5)
    assert result[0].persistence_count == 5, (
        f"Expected persistence_count=5 (4+1), got {result[0].persistence_count}"
    )


def test_new_blob_gets_new_id() -> None:
    """An unmatched new blob must receive a fresh blob_id and persistence_count=1."""
    prev = (make_blob(blob_id=5, bbox=(100, 100, 120, 120), persistence_count=3),)
    # New blob far away — no IoU match possible
    new_far = (make_blob(blob_id=99, bbox=(0, 0, 10, 10), persistence_count=1),)
    result = match_blobs(prev, new_far, iou_threshold=0.5)
    assert len(result) == 1
    # Unmatched blob must NOT have blob_id=5 (the previous blob's ID)
    assert result[0].blob_id != 5, (
        f"Unmatched blob incorrectly inherited ID=5 from the previous blob"
    )
    assert result[0].persistence_count == 1, (
        f"Unmatched new blob should have persistence_count=1, got {result[0].persistence_count}"
    )


def test_match_blobs_empty_prev() -> None:
    """With no previous blobs, all new blobs get fresh IDs and persistence_count=1."""
    new = (
        make_blob(blob_id=99, bbox=(0, 0, 10, 10)),
        make_blob(blob_id=100, bbox=(50, 50, 60, 60)),
    )
    result = match_blobs((), new, iou_threshold=0.5)
    assert len(result) == 2
    for blob in result:
        assert blob.persistence_count == 1


def test_match_blobs_empty_new() -> None:
    """With no new blobs, all previous blobs are dropped. Result is empty tuple."""
    prev = (make_blob(blob_id=1), make_blob(blob_id=2))
    result = match_blobs(prev, (), iou_threshold=0.5)
    assert result == ()


def test_match_blobs_unmatched_prev_dropped() -> None:
    """Previous blobs with no IoU match to any new blob must be dropped from the result."""
    prev = (
        make_blob(blob_id=1, bbox=(0, 0, 10, 10)),   # will match
        make_blob(blob_id=2, bbox=(200, 200, 210, 210)),  # far away, no match
    )
    new = (make_blob(blob_id=99, bbox=(0, 0, 10, 10)),)  # matches blob_id=1 only
    result = match_blobs(prev, new, iou_threshold=0.5)
    assert len(result) == 1
    assert result[0].blob_id == 1
