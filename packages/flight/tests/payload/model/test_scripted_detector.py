"""Tests for the scripted detector backend, including classifier skip."""

import importlib.util

import numpy as np
import pytest
from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import MessageType, Ok
from flight.payload.model import DetectorBackend, OnnxDetector, ScriptedDetector


def _processed_frame(height: int = 20, width: int = 20) -> ProcessedFrameMsg:
    """Build a minimal ProcessedFrameMsg (tensor content is unused by ScriptedDetector)."""
    tensor = np.zeros((4, height, width), dtype=np.float32)  # np.ndarray[float32, (C, H, W)]
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
    assert result.value.inference_ms == 0.0


def test_scripted_detector_satisfies_protocol() -> None:
    """ScriptedDetector conforms to DetectorBackend (typed + runtime)."""
    detector: DetectorBackend = ScriptedDetector(np.zeros((4, 4), dtype=np.float32))
    assert isinstance(detector, DetectorBackend)


def test_negative_classifier_skips_segmentor() -> None:
    """A negative scripted classifier returns empty blobs and a zero mask."""
    mask = np.ones((20, 20), dtype=np.float32)
    detector = ScriptedDetector(mask, classifier_positive=False)
    result = detector.detect(_processed_frame())
    assert isinstance(result, Ok)
    assert result.value.blobs == ()
    assert float(np.asarray(result.value.mask).max()) == 0.0


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is not None,
    reason="onnxruntime is installed; the absent-runtime guard cannot be exercised",
)
def test_onnx_detector_requires_onnxruntime_when_absent() -> None:
    """Constructing OnnxDetector without onnxruntime raises ImportError."""
    with pytest.raises(ImportError):
        OnnxDetector("segmentor.onnx", "classifier.onnx")
