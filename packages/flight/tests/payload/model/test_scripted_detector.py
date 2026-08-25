"""Tests for the scripted detector backend."""

import numpy as np
from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import MessageType, Ok
from flight.payload.model import DetectorBackend, ScriptedDetector


def _processed_frame() -> ProcessedFrameMsg:
    """Build a minimal ProcessedFrameMsg (tensor content is unused by ScriptedDetector)."""
    tensor = np.zeros((4, 20, 20), dtype=np.float32)  # np.ndarray[float32, (C, H, W)]
    return ProcessedFrameMsg(
        msg_type=MessageType.PROCESSED_FRAME,
        timestamp_utc="2026-05-31T00:00:00.000Z",
        frame_id=7,
        tensor=tensor,
        quality_flags=frozenset(),
        crop_origin_px=(0, 0),
        scale_factor=1.0,
    )


def test_scripted_detector_returns_blobs() -> None:
    """ScriptedDetector returns Ok(InferenceResultMsg) with blobs from its mask."""
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[2:8, 2:8] = 1.0
    detector = ScriptedDetector(mask, confidence_gate=0.5, min_blob_area_px=4)
    result = detector.detect(_processed_frame())
    assert isinstance(result, Ok)
    assert result.value.frame_id == 7
    assert len(result.value.blobs) == 1
    assert result.value.model_version == "scripted"


def test_scripted_detector_satisfies_protocol() -> None:
    """ScriptedDetector conforms to DetectorBackend (typed + runtime)."""
    detector: DetectorBackend = ScriptedDetector(np.zeros((4, 4), dtype=np.float32))
    assert isinstance(detector, DetectorBackend)
