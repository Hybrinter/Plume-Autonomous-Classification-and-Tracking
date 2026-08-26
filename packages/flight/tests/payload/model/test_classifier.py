"""Tests for the scripted classifier backend."""

import importlib.util

import numpy as np
import pytest
from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import MessageType, Ok
from flight.payload.model import ClassifierBackend, OnnxClassifier, ScriptedClassifier


def _processed_frame() -> ProcessedFrameMsg:
    """Build a dummy processed frame. ScriptedClassifier ignores the tensor."""
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


def test_scripted_classifier_default_is_positive() -> None:
    """Default ScriptedClassifier is always-positive."""
    result = ScriptedClassifier().classify(_processed_frame())
    assert isinstance(result, Ok)
    assert result.value.positive is True


def test_scripted_classifier_can_be_negative() -> None:
    """ScriptedClassifier(positive=False) reports a negative decision."""
    result = ScriptedClassifier(positive=False, logit=-2.0).classify(_processed_frame())
    assert isinstance(result, Ok)
    assert result.value.positive is False
    assert result.value.logit == -2.0


def test_scripted_classifier_satisfies_protocol() -> None:
    """ScriptedClassifier conforms to ClassifierBackend."""
    classifier: ClassifierBackend = ScriptedClassifier()
    assert isinstance(classifier, ClassifierBackend)


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is not None,
    reason="onnxruntime is installed; the absent-runtime guard cannot be exercised",
)
def test_onnx_classifier_requires_onnxruntime_when_absent() -> None:
    """Constructing OnnxClassifier without onnxruntime raises ImportError."""
    with pytest.raises(ImportError):
        OnnxClassifier("classifier.onnx")
