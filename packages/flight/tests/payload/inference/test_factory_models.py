"""Factory ONNX artifacts under data/models/ load through the flight backends."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import MessageType, Ok
from flight.payload.inference import OnnxDetector, compute_sha256

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
def test_factory_pair_loads_at_train_export_size() -> None:
    """Shipped factory ONNX is 256. Flight inference input is 512 until retraining."""
    from flight.libs.config import PactConfig

    flight = PactConfig().inference
    assert flight.input_height_px == 512
    assert flight.input_width_px == 512
    cls_manifest = _manifest(_CLASSIFIER)
    seg_manifest = _manifest(_SEGMENTOR)
    assert cls_manifest["input_shape"] == [1, 4, 256, 256]
    assert cls_manifest["output_shape"] == [1, 1]
    assert seg_manifest["input_shape"] == [1, 4, 256, 256]
    assert seg_manifest["output_shape"] == [1, 1, 256, 256]
    assert cls_manifest["quantization"] == "fp32"
    assert seg_manifest["quantization"] == "fp32"
    assert compute_sha256(str(_CLASSIFIER)) == cls_manifest["sha256"]
    assert compute_sha256(str(_SEGMENTOR)) == seg_manifest["sha256"]
    detector = OnnxDetector(
        str(_SEGMENTOR),
        str(_CLASSIFIER),
        confidence_gate=0.55,
        min_blob_area_px=15,
        logit_threshold=0.0,
        classifier_sha256=str(cls_manifest["sha256"]),
        segmentor_sha256=str(seg_manifest["sha256"]),
        expected_input_shape=(1, 4, 256, 256),
        expected_segmentor_output_shape=(1, 1, 256, 256),
        expected_classifier_output_shape=(1, 1),
    )
    frame = ProcessedFrameMsg(
        msg_type=MessageType.PROCESSED_FRAME,
        timestamp_utc="2026-08-30T00:00:00.000Z",
        frame_id=0,
        tensor=np.zeros((4, 256, 256), dtype=np.float32),
        quality_flags=frozenset(),
        crop_origin_px=(0, 0),
        scale_factor=1.0,
    )
    result = detector.detect(frame)
    assert isinstance(result, Ok)
    assert result.value.blobs == ()
