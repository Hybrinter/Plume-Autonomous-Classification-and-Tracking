"""Tests for the pure blob-extraction function."""

import numpy as np
from flight.payload.model.blobs import extract_blobs


def test_extracts_two_blobs() -> None:
    """Two separated high-confidence regions yield two blobs."""
    mask = np.zeros((20, 20), dtype=np.float32)  # np.ndarray[float32, (H, W)]
    mask[2:6, 2:6] = 1.0
    mask[12:18, 12:18] = 1.0
    blobs = extract_blobs(mask, confidence_gate=0.5, min_blob_area_px=4)
    assert len(blobs) == 2
    areas = sorted(blob.pixel_area for blob in blobs)
    assert areas == [16, 36]


def test_below_min_area_excluded() -> None:
    """A region smaller than min_blob_area_px is dropped."""
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[5:7, 5:7] = 1.0  # area 4
    blobs = extract_blobs(mask, confidence_gate=0.5, min_blob_area_px=10)
    assert blobs == ()


def test_bbox_and_centroid() -> None:
    """A single square blob has the expected bbox and centroid."""
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:5, 3:6] = 1.0  # x in [3,5], y in [2,4]
    blobs = extract_blobs(mask, confidence_gate=0.5, min_blob_area_px=1)
    assert len(blobs) == 1
    blob = blobs[0]
    assert blob.bbox == (3, 2, 5, 4)
    assert blob.centroid_raw == (4.0, 3.0)
