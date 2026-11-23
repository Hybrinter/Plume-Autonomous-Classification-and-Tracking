"""Legacy factory ONNX artifacts retain truthful metadata and fail the new contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from flight.payload.inference import compute_sha256
from flight.payload.inference.onnx_session import load_onnx_session

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="onnxruntime extra not installed",
)

_REPO = Path(__file__).resolve().parents[5]
_CLASSIFIER = _REPO / "data" / "models" / "active_classifier.onnx"
_SEGMENTOR = _REPO / "data" / "models" / "active_segmentor.onnx"


def _manifest(artifact: Path) -> dict[str, object]:
    """Load the JSON sidecar next to an ONNX artifact."""
    payload = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {artifact.with_suffix('.json')}")
    return {str(key): value for key, value in payload.items()}


@pytest.mark.skipif(
    not _CLASSIFIER.is_file() or not _SEGMENTOR.is_file(), reason="factory ONNX absent"
)
def test_factory_artifacts_are_explicitly_legacy_and_rejected() -> None:
    """Factory binaries keep their old metadata and cannot satisfy conditioned flight loading."""
    cls_manifest = _manifest(_CLASSIFIER)
    seg_manifest = _manifest(_SEGMENTOR)
    assert cls_manifest["input_shape"] == [1, 4, 1024, 1224]
    assert cls_manifest["output_shape"] == [1, 1]
    assert seg_manifest["input_shape"] == [1, 4, 1024, 1224]
    assert seg_manifest["output_shape"] == [1, 1, 1024, 1224]
    assert cls_manifest["quantization"] == "fp16"
    assert seg_manifest["quantization"] == "int8"
    assert compute_sha256(str(_CLASSIFIER)) == cls_manifest["sha256"]
    assert compute_sha256(str(_SEGMENTOR)) == seg_manifest["sha256"]
    with pytest.raises(ValueError, match="image and gsd inputs"):
        load_onnx_session(str(_CLASSIFIER))
    with pytest.raises(ValueError, match="image and gsd inputs"):
        load_onnx_session(str(_SEGMENTOR))
