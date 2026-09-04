"""Tests for INT8 sibling path resolution."""

from pathlib import Path

from flight.payload.inference.artifact_path import int8_sibling_path, resolve_quantized_path


def test_int8_sibling_path_uses_stem() -> None:
    """INT8 sibling sits next to the FP32 file as ``<stem>.int8.onnx``."""
    assert int8_sibling_path("data/models/active_classifier.onnx") == Path(
        "data/models/active_classifier.int8.onnx"
    )


def test_resolve_quantized_path_false_keeps_fp32() -> None:
    """``use_int8=false`` returns the configured FP32 path."""
    path = "data/models/active_segmentor.onnx"
    assert resolve_quantized_path(path, use_int8=False) == path


def test_resolve_quantized_path_true_returns_sibling() -> None:
    """``use_int8=true`` returns the INT8 sibling path without checking the file."""
    assert resolve_quantized_path("data/models/active_segmentor.onnx", use_int8=True) == str(
        Path("data/models/active_segmentor.int8.onnx")
    )
