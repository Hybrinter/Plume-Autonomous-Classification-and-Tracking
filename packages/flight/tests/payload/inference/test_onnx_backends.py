"""Dynamic image-plus-GSD ONNX classifier and segmentor contract tests."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import Err, MessageType, Ok
from flight.payload.inference import OnnxClassifier, OnnxDetector, OnnxSegmentor

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None or importlib.util.find_spec("onnx") is None,
    reason="onnx and onnxruntime extras not installed",
)


def _save_model(path: Path, kind: str) -> None:
    """Build a tiny dynamic-batch graph whose outputs depend on image and GSD."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [None, 4, 8, 8])
    gsd = helper.make_tensor_value_info("gsd", TensorProto.FLOAT, [None, 2])
    if kind == "classifier":
        output = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 1])
        initializers = [
            numpy_helper.from_array(np.array(0.1, dtype=np.float32), "gsd_scale"),
            numpy_helper.from_array(np.array(-0.1, dtype=np.float32), "bias"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "unsqueeze_axis"),
        ]
        nodes = [
            helper.make_node("ReduceMean", ["image"], ["image_mean"], axes=[1, 2, 3], keepdims=0),
            helper.make_node("ReduceMean", ["gsd"], ["gsd_mean"], axes=[1], keepdims=0),
            helper.make_node("Mul", ["gsd_mean", "gsd_scale"], ["gsd_term"]),
            helper.make_node("Add", ["image_mean", "gsd_term"], ["combined"]),
            helper.make_node("Add", ["combined", "bias"], ["raw_logit"]),
            helper.make_node("Unsqueeze", ["raw_logit", "unsqueeze_axis"], ["logits"]),
        ]
    elif kind == "segmentor":
        output = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 1, 8, 8])
        initializers = [
            numpy_helper.from_array(np.array(4.0, dtype=np.float32), "image_scale"),
            numpy_helper.from_array(np.array(-1.0, dtype=np.float32), "bias"),
            numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64), "unsqueeze_axes"),
        ]
        nodes = [
            helper.make_node("ReduceMean", ["image"], ["image_mean"], axes=[1], keepdims=1),
            helper.make_node("Mul", ["image_mean", "image_scale"], ["image_term"]),
            helper.make_node("ReduceMean", ["gsd"], ["gsd_mean"], axes=[1], keepdims=0),
            helper.make_node("Unsqueeze", ["gsd_mean", "unsqueeze_axes"], ["gsd_broadcast"]),
            helper.make_node("Add", ["image_term", "bias"], ["biased"]),
            helper.make_node("Add", ["biased", "gsd_broadcast"], ["logits"]),
        ]
    else:
        raise ValueError(kind)
    graph = helper.make_graph(nodes, f"dynamic_{kind}", [image, gsd], [output], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


@pytest.fixture
def models(tmp_path: Path) -> tuple[Path, Path]:
    classifier_path = tmp_path / "classifier.onnx"
    segmentor_path = tmp_path / "segmentor.onnx"
    _save_model(classifier_path, "classifier")
    _save_model(segmentor_path, "segmentor")
    return classifier_path, segmentor_path


def _frame(tensor: np.ndarray, tile_gsd_m: np.ndarray | None = None) -> ProcessedFrameMsg:
    """Build a frame with full-image NCHW input and optional actual tile GSD."""
    return ProcessedFrameMsg(
        msg_type=MessageType.PROCESSED_FRAME,
        timestamp_utc="2026-10-01T00:00:00.000Z",
        frame_id=3,
        tensor=tensor,
        quality_flags=frozenset(),
        tile_gsd_m=tile_gsd_m,
    )


def test_onnx_classifier_supports_dynamic_batches_and_uses_gsd(
    models: tuple[Path, Path],
) -> None:
    classifier_path, _ = models
    classifier = OnnxClassifier(str(classifier_path), logit_threshold=0.0)
    for count in (1, 3, 64):
        images = np.zeros((count, 4, 8, 8), dtype=np.float32)
        encoded_gsd = np.zeros((count, 2), dtype=np.float32)
        result = classifier.classify_tiles(images, encoded_gsd)
        assert isinstance(result, Ok) and result.value.shape == (count,)
    image = np.zeros((1, 4, 8, 8), dtype=np.float32)
    baseline = classifier.classify_tiles(image, np.zeros((1, 2), dtype=np.float32))
    changed_gsd = classifier.classify_tiles(image, np.ones((1, 2), dtype=np.float32))
    assert isinstance(baseline, Ok) and isinstance(changed_gsd, Ok)
    assert baseline.value[0] != changed_gsd.value[0]


def test_onnx_segmentor_supports_dynamic_batches_and_uses_gsd(
    models: tuple[Path, Path],
) -> None:
    _, segmentor_path = models
    segmentor = OnnxSegmentor(str(segmentor_path))
    for count in (1, 3, 64):
        images = np.ones((count, 4, 8, 8), dtype=np.float32)
        result = segmentor.segment_tiles(images, np.zeros((count, 2), dtype=np.float32))
        assert isinstance(result, Ok) and result.value.shape == (count, 1, 8, 8)
    image = np.zeros((1, 4, 8, 8), dtype=np.float32)
    baseline = segmentor.segment_tiles(image, np.zeros((1, 2), dtype=np.float32))
    changed_gsd = segmentor.segment_tiles(image, np.ones((1, 2), dtype=np.float32))
    assert isinstance(baseline, Ok) and isinstance(changed_gsd, Ok)
    assert not np.array_equal(baseline.value, changed_gsd.value)


def test_onnx_detector_uses_new_tile_interface(models: tuple[Path, Path]) -> None:
    classifier_path, segmentor_path = models
    detector = OnnxDetector(
        str(segmentor_path),
        str(classifier_path),
        confidence_gate=0.55,
        min_blob_area_px=4,
        grid=(1, 1),
        expected_input_shape=(None, 4, 8, 8),
        expected_classifier_output_shape=(None, 1),
        expected_segmentor_output_shape=(None, 1, 8, 8),
    )
    tile_gsd = np.full((1, 2), 15.87, dtype=np.float32)
    empty = detector.detect(_frame(np.zeros((1, 4, 8, 8), dtype=np.float32), tile_gsd))
    assert isinstance(empty, Ok)
    assert empty.value.tile_positive == (False,)
    assert float(np.asarray(empty.value.mask).max()) == 0.0

    blob = np.zeros((1, 4, 8, 8), dtype=np.float32)
    blob[:, :, 2:6, 2:6] = 1
    result = detector.detect(_frame(blob, tile_gsd))
    assert isinstance(result, Ok)
    assert result.value.tile_positive == (True,)
    assert float(np.asarray(result.value.mask).max()) > 0.7
    assert len(result.value.blobs) == 1


def test_onnx_backends_reject_nonfinite_and_malformed_arrays(
    models: tuple[Path, Path],
) -> None:
    classifier_path, segmentor_path = models
    classifier = OnnxClassifier(str(classifier_path))
    segmentor = OnnxSegmentor(str(segmentor_path))
    malformed = classifier.classify_tiles(
        np.ones((2, 4, 8, 8), dtype=np.float32), np.zeros((1, 2), dtype=np.float32)
    )
    assert isinstance(malformed, Err)
    nonfinite = segmentor.segment_tiles(
        np.full((1, 4, 8, 8), np.nan, dtype=np.float32), np.zeros((1, 2), dtype=np.float32)
    )
    assert isinstance(nonfinite, Err)
