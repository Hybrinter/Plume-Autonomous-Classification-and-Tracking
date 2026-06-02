"""Tests for SIL plume scene generation."""

import numpy as np
from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import MessageType, Ok
from sim.scene import build_frames, plume_detector


def test_build_frames_count_and_shape() -> None:
    """build_frames returns N frames each shaped (4, 256, 256)."""
    frames = build_frames(3)
    assert len(frames) == 3
    assert np.asarray(frames[0].raw_bands).shape == (4, 256, 256)
    assert frames[0].frame_id == 1
    assert frames[2].frame_id == 3


def test_plume_detector_finds_one_blob() -> None:
    """The scripted plume detector yields exactly one blob on a processed frame."""
    detector = plume_detector()
    frame = ProcessedFrameMsg(
        msg_type=MessageType.PROCESSED_FRAME,
        timestamp_utc="t",
        frame_id=1,
        tensor=np.zeros((4, 256, 256), dtype=np.float32),
        quality_flags=frozenset(),
        crop_origin_px=(0, 0),
        scale_factor=1.0,
    )
    result = detector.detect(frame)
    assert isinstance(result, Ok)
    assert len(result.value.blobs) == 1
