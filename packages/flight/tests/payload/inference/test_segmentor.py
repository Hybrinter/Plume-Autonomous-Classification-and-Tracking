"""Tests for the scripted segmentor backend."""

import importlib.util

import numpy as np
import pytest
from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import MessageType, Ok
from flight.payload.inference import OnnxSegmentor, ScriptedSegmentor, SegmentorBackend


def _processed_frame() -> ProcessedFrameMsg:
    """Build a dummy processed frame. ScriptedSegmentor ignores the tensor."""
    tensor = np.zeros((4, 8, 8), dtype=np.float32)  # np.ndarray[float32, (C, H, W)]
    return ProcessedFrameMsg(
        msg_type=MessageType.PROCESSED_FRAME,
        timestamp_utc="2026-05-31T00:00:00.000Z",
        frame_id=1,
        tensor=tensor,
        quality_flags=frozenset(),
        crop_origin_px=(0, 0),
        scale_factor=1.0,
    )


def test_scripted_segmentor_returns_configured_mask() -> None:
    """ScriptedSegmentor returns the mask it was constructed with."""
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[1:3, 1:3] = 0.9
    result = ScriptedSegmentor(mask).segment(_processed_frame())
    assert isinstance(result, Ok)
    np.testing.assert_array_equal(result.value, mask)


def test_scripted_segmentor_satisfies_protocol() -> None:
    """ScriptedSegmentor conforms to SegmentorBackend."""
    segmentor: SegmentorBackend = ScriptedSegmentor(np.zeros((4, 4), dtype=np.float32))
    assert isinstance(segmentor, SegmentorBackend)


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is not None,
    reason="onnxruntime is installed; the absent-runtime guard cannot be exercised",
)
def test_onnx_segmentor_requires_onnxruntime_when_absent() -> None:
    """Constructing OnnxSegmentor without onnxruntime raises ImportError."""
    with pytest.raises(ImportError):
        OnnxSegmentor("segmentor.onnx")
