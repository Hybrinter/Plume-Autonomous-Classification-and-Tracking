"""Live ONNX classifier/segmentor/detector tests (skip without onnxruntime)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import MessageType, Ok
from flight.payload.inference import OnnxClassifier, OnnxDetector, OnnxSegmentor

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="onnxruntime extra not installed",
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_CLASSIFIER = _FIXTURES / "tiny_classifier.onnx"
_SEGMENTOR = _FIXTURES / "tiny_segmentor.onnx"


def _frame(tensor: np.ndarray) -> ProcessedFrameMsg:
    """Build a processed frame around a (C, H, W) tensor."""
    return ProcessedFrameMsg(
        msg_type=MessageType.PROCESSED_FRAME,
        timestamp_utc="2026-05-31T00:00:00.000Z",
        frame_id=3,
        tensor=tensor,
        quality_flags=frozenset(),
        crop_origin_px=(0, 0),
        scale_factor=1.0,
    )


def _blob_tensor() -> np.ndarray:
    """(4, 8, 8) tensor with a 4x4 square of ones."""
    tensor = np.zeros((4, 8, 8), dtype=np.float32)
    tensor[:, 2:6, 2:6] = 1.0
    return tensor


def test_onnx_classifier_positive_on_blob() -> None:
    """Tiny classifier is negative on zeros and positive on a planted blob."""
    clf = OnnxClassifier(str(_CLASSIFIER), logit_threshold=0.0)
    empty = clf.classify(_frame(np.zeros((4, 8, 8), dtype=np.float32)))
    blob = clf.classify(_frame(_blob_tensor()))
    assert isinstance(empty, Ok) and empty.value.positive is False
    assert isinstance(blob, Ok) and blob.value.positive is True


def test_onnx_segmentor_blob_mask() -> None:
    """Tiny segmentor lights up the planted square after sigmoid."""
    seg = OnnxSegmentor(str(_SEGMENTOR))
    result = seg.segment(_frame(_blob_tensor()))
    assert isinstance(result, Ok)
    mask = result.value
    assert mask.shape == (8, 8)
    assert float(mask[4, 4]) > 0.7
    assert float(mask[0, 0]) < 0.3


def test_onnx_detector_skips_segmentor_on_empty() -> None:
    """OnnxDetector returns empty blobs on an all-zero frame (negative class)."""
    detector = OnnxDetector(
        str(_SEGMENTOR),
        str(_CLASSIFIER),
        confidence_gate=0.55,
        min_blob_area_px=4,
    )
    result = detector.detect(_frame(np.zeros((4, 8, 8), dtype=np.float32)))
    assert isinstance(result, Ok)
    assert result.value.blobs == ()
    assert float(np.asarray(result.value.mask).max()) == 0.0


def test_onnx_detector_extracts_blob_on_positive() -> None:
    """OnnxDetector runs the segmentor on a positive frame and extracts a blob."""
    detector = OnnxDetector(
        str(_SEGMENTOR),
        str(_CLASSIFIER),
        confidence_gate=0.55,
        min_blob_area_px=4,
    )
    result = detector.detect(_frame(_blob_tensor()))
    assert isinstance(result, Ok)
    assert len(result.value.blobs) == 1
    assert float(np.asarray(result.value.mask).max()) > 0.7
