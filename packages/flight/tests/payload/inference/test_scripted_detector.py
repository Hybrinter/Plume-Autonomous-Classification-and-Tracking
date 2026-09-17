"""Tests for the scripted detector backend, including classifier skip."""

import importlib.util

import numpy as np
import pytest
from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import FrameUsabilityTag, MessageType, Ok
from flight.payload.inference import DetectorBackend, OnnxDetector, ScriptedDetector


def _processed_frame(height: int = 20, width: int = 20) -> ProcessedFrameMsg:
    """Build a minimal ProcessedFrameMsg (tensor content is unused by ScriptedDetector)."""
    tensor = np.zeros((4, height, width), dtype=np.float32)  # np.ndarray[float32, (C, H, W)]
    return ProcessedFrameMsg(
        msg_type=MessageType.PROCESSED_FRAME,
        timestamp_utc="2026-05-31T00:00:00.000Z",
        frame_id=7,
        tensor=tensor,
        quality_flags=frozenset(),
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


def test_scripted_detector_load_mask_bootstrap() -> None:
    """load_mask replaces the constructor mask before the next detect()."""
    detector = ScriptedDetector(np.zeros((20, 20), dtype=np.float32), min_blob_area_px=4)
    empty = detector.detect(_processed_frame())
    assert isinstance(empty, Ok)
    assert empty.value.blobs == ()
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[2:8, 2:8] = 1.0
    detector.load_mask(mask)
    result = detector.detect(_processed_frame())
    assert isinstance(result, Ok)
    assert len(result.value.blobs) == 1


def test_negative_classifier_skips_segmentor() -> None:
    """A negative scripted classifier returns empty blobs and a zero mask."""
    mask = np.ones((20, 20), dtype=np.float32)
    detector = ScriptedDetector(mask, classifier_positive=False)
    result = detector.detect(_processed_frame())
    assert isinstance(result, Ok)
    assert result.value.blobs == ()
    assert float(np.asarray(result.value.mask).max()) == 0.0


def test_detector_preserves_quality_for_science_qualification() -> None:
    """Image quality follows inference without becoming a control safety flag."""
    frame = _processed_frame()
    frame = ProcessedFrameMsg(
        msg_type=frame.msg_type,
        timestamp_utc=frame.timestamp_utc,
        frame_id=frame.frame_id,
        tensor=frame.tensor,
        quality_flags=frozenset({FrameUsabilityTag.MOTION_SMEAR}),
    )
    result = ScriptedDetector(np.zeros((20, 20), dtype=np.float32)).detect(frame)
    assert isinstance(result, Ok)
    assert result.value.quality_flags == frozenset({FrameUsabilityTag.MOTION_SMEAR})
    assert result.value.mode_flags == 0


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is not None,
    reason="onnxruntime is installed; the absent-runtime guard cannot be exercised",
)
def test_onnx_detector_requires_onnxruntime_when_absent() -> None:
    """Constructing OnnxDetector without onnxruntime raises ImportError."""
    with pytest.raises(ImportError):
        OnnxDetector("segmentor.onnx", "classifier.onnx")
