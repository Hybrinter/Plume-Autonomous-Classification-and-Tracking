"""Unit tests for flight.payload.gimbal.safety -- confidence and area gates.

REQ-AIML-DATA-008, REQ-AIML-DATA-009, REQ-AIML-GIMB-006, REQ-AIML-GIMB-007
"""

import pytest
from flight.libs.messages import BlobMeta
from flight.payload.gimbal import apply_confidence_gate, apply_min_area_gate


def make_blob(
    blob_id: int = 1,
    mean_confidence: float = 0.85,
    pixel_area: int = 200,
) -> BlobMeta:
    """Construct a BlobMeta for safety gate tests."""
    return BlobMeta(
        blob_id=blob_id,
        bbox=(0, 0, 20, 20),
        centroid_raw=(10.0, 10.0),
        pixel_area=pixel_area,
        mean_confidence=mean_confidence,
        persistence_count=1,
    )


@pytest.mark.parametrize(
    "confidence,expected_count",
    [
        (0.30, 0),
        (0.54, 0),
        (0.55, 1),
        (0.90, 1),
    ],
)
def test_confidence_gate_filters_low_confidence(
    confidence: float,
    expected_count: int,
) -> None:
    """Blobs with mean_confidence < 0.55 must be rejected by apply_confidence_gate."""
    blobs = (make_blob(mean_confidence=confidence),)
    result = apply_confidence_gate(blobs, threshold=0.55)
    assert len(result) == expected_count


def test_confidence_gate_passes_all_above_threshold() -> None:
    """All blobs above threshold must pass through."""
    blobs = (
        make_blob(blob_id=1, mean_confidence=0.60),
        make_blob(blob_id=2, mean_confidence=0.80),
        make_blob(blob_id=3, mean_confidence=1.00),
    )
    result = apply_confidence_gate(blobs, threshold=0.55)
    assert len(result) == 3


def test_confidence_gate_empty_input() -> None:
    """Empty input tuple must return empty output tuple."""
    result = apply_confidence_gate((), threshold=0.55)
    assert result == ()


def test_min_area_gate_filters_small_blobs() -> None:
    """Blobs with pixel_area < min_px must be rejected."""
    small = make_blob(blob_id=1, pixel_area=10)
    large = make_blob(blob_id=2, pixel_area=100)
    result = apply_min_area_gate((small, large), min_px=15)
    assert len(result) == 1
    assert result[0].blob_id == 2


@pytest.mark.parametrize(
    "area,min_px,expected_count",
    [
        (5, 15, 0),
        (14, 15, 0),
        (15, 15, 1),
        (100, 15, 1),
    ],
)
def test_min_area_gate_boundary(area: int, min_px: int, expected_count: int) -> None:
    """Parametrized boundary test for apply_min_area_gate."""
    blobs = (make_blob(pixel_area=area),)
    result = apply_min_area_gate(blobs, min_px=min_px)
    assert len(result) == expected_count
