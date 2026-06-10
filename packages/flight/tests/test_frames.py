"""Tests for the MosaicFrame raw-frame value type."""

import numpy as np
from flight.libs.types import MosaicFrame


def test_mosaic_frame_holds_uint16_plane() -> None:
    """MosaicFrame carries the raw mosaic plane and capture metadata."""
    mosaic = np.zeros((4, 4), dtype=np.uint16)  # np.ndarray[uint16, (H, W)]
    frame = MosaicFrame(
        timestamp_utc="2026-06-09T00:00:00.000Z",
        frame_id=1,
        mosaic=mosaic,
        exposure_us=1000.0,
        gain_db=0.0,
    )
    assert frame.frame_id == 1
    assert np.asarray(frame.mosaic).dtype == np.uint16
