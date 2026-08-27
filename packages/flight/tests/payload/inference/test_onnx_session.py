"""Tests for the lazy onnxruntime session loader."""

import importlib.util

import pytest
from flight.payload.inference.onnx_session import load_onnx_session, onnx_tensor_shape


def test_onnx_tensor_shape_maps_symbolic_dims() -> None:
    """Non-integer dims become None; integers are kept."""
    assert onnx_tensor_shape([1, "N", 256]) == (1, None, 256)


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is not None,
    reason="onnxruntime is installed; the absent-runtime guard cannot be exercised",
)
def test_load_onnx_session_requires_onnxruntime_when_absent() -> None:
    """load_onnx_session without onnxruntime raises ImportError."""
    with pytest.raises(ImportError):
        load_onnx_session("model.onnx")
