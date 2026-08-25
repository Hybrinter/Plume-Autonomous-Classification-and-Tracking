"""Tests for the ONNX detector backend's optional-dependency guard."""

import importlib.util

import pytest
from flight.payload.model import OnnxDetector


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is not None,
    reason="onnxruntime is installed; the absent-runtime guard cannot be exercised",
)
def test_onnx_detector_requires_onnxruntime_when_absent() -> None:
    """Constructing OnnxDetector without onnxruntime raises a helpful ImportError."""
    with pytest.raises(ImportError):
        OnnxDetector("model.onnx")
