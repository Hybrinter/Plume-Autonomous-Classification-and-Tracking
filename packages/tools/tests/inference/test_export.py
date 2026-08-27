"""ONNX export, manifest, INT8 sibling, and promote tests."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tools.inference.accept import (
    GoldenClassifierScene,
    GoldenScene,
    Manifest,
    accept_artifact,
    accept_classifier_artifact,
    load_manifest,
)
from tools.inference.data import make_synthetic_pack, write_processed_pack
from tools.inference.export import (
    ExportConfig,
    export,
    int8_artifact_path,
    promote,
    write_manifest,
)
from tools.inference.split import SplitRecipe
from tools.inference.train import TrainConfig, train

_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None
_skip_no_onnx = pytest.mark.skipif(not _HAS_ONNX, reason="onnx extra not installed")
_skip_no_ort = pytest.mark.skipif(not _HAS_ORT, reason="onnxruntime extra not installed")


def _gold_mask_for(tensor: torch.Tensor, scenes: list[GoldenScene]) -> torch.Tensor:
    """Return the gold mask of the matching scene (perfect predictor)."""
    for scene in scenes:
        if scene.input_tensor is tensor:
            return scene.gold_mask
    return scenes[0].gold_mask


def test_int8_artifact_path() -> None:
    """INT8 sibling uses the FP32 stem plus `.int8.onnx`."""
    assert int8_artifact_path("/tmp/seg.onnx") == Path("/tmp/seg.int8.onnx")


def test_write_manifest_includes_quantization(tmp_path: Path) -> None:
    """write_manifest persists the quantization field."""
    path = tmp_path / "seg.json"
    write_manifest(
        str(path),
        Manifest(
            version="v1",
            model_repo_sha="abc",
            dataset_hash="ds",
            input_shape=(1, 4, 32, 32),
            output_shape=(1, 1, 32, 32),
            sha256="0" * 64,
            quantization="int8",
        ),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["quantization"] == "int8"
    assert load_manifest(str(path)).quantization == "int8"


def test_write_manifest_defaults_fp32(tmp_path: Path) -> None:
    """Manifest without an explicit field serializes as fp32."""
    path = tmp_path / "seg.json"
    write_manifest(
        str(path),
        Manifest(
            version="v1",
            model_repo_sha="abc",
            dataset_hash="ds",
            input_shape=(1, 4, 32, 32),
            output_shape=(1, 1),
            sha256="0" * 64,
        ),
    )
    assert json.loads(path.read_text(encoding="utf-8"))["quantization"] == "fp32"


def test_calibration_batches_random() -> None:
    """Empty calib_dir yields synthetic [0, 1] NCHW tensors."""
    from tools.inference.export import _calibration_batches

    batches = _calibration_batches((1, 4, 8, 8), "", 3)
    assert len(batches) == 3
    assert batches[0].shape == (1, 4, 8, 8)
    assert batches[0].dtype == np.float32
    assert float(batches[0].min()) >= 0.0
    assert float(batches[0].max()) <= 1.0


def test_calibration_batches_from_pack(tmp_path: Path) -> None:
    """A processed pack supplies train-split calibration tensors."""
    from tools.inference.export import _calibration_batches

    images, masks, labels = make_synthetic_pack(6, 4, 8, 8, seed=0)
    write_processed_pack(tmp_path, images, masks, labels, SplitRecipe())
    batches = _calibration_batches((1, 4, 8, 8), str(tmp_path), 2)
    assert len(batches) == 2
    assert batches[0].shape == (1, 4, 8, 8)


@_skip_no_onnx
def test_export_segmentor_then_accept(tmp_path: Path) -> None:
    """1-step synthetic 256 segmentor exports ONNX logits and passes injected accept."""
    ckpt = tmp_path / "seg.pt"
    train(
        TrainConfig(
            kind="segmentor",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            checkpoint_path=str(ckpt),
            run_dir=str(tmp_path / "runs"),
            seed=0,
        )
    )
    onnx_path, manifest_path, manifest = export(
        ExportConfig(
            kind="segmentor",
            checkpoint_path=str(ckpt),
            output_path=str(tmp_path / "seg.onnx"),
        )
    )
    assert onnx_path.is_file()
    assert manifest_path.is_file()
    assert manifest.input_shape == (1, 4, 256, 256)
    assert manifest.output_shape == (1, 1, 256, 256)
    assert manifest.quantization == "fp32"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["quantization"] == "fp32"

    tensor = torch.zeros((4, 256, 256), dtype=torch.float32)
    gold = torch.zeros((256, 256), dtype=torch.float32)
    gold[64:192, 64:192] = 1.0
    scenes = [GoldenScene(input_tensor=tensor, gold_mask=gold)]
    report = accept_artifact(
        str(onnx_path),
        manifest,
        scenes,
        run_inference=lambda t: _gold_mask_for(t, scenes),
        expected_input=(1, 4, 256, 256),
        expected_output=(1, 1, 256, 256),
        min_iou=0.9,
        max_latency_ms=10_000.0,
    )
    assert report.accepted
    dest = promote(str(onnx_path), str(tmp_path / "active_segmentor.onnx"), report)
    assert dest.is_file()
    assert dest.with_suffix(".json").is_file()
    if importlib.util.find_spec("onnxruntime") is not None:
        from tools.inference.accept import onnx_inference_fn

        pred = onnx_inference_fn(str(onnx_path))(tensor)
        assert pred.shape == (256, 256)


@_skip_no_onnx
def test_export_classifier_then_accept(tmp_path: Path) -> None:
    """1-step synthetic 256 classifier exports (1, 1) logits and passes injected accept."""
    ckpt = tmp_path / "clf.pt"
    train(
        TrainConfig(
            kind="classifier",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            checkpoint_path=str(ckpt),
            run_dir=str(tmp_path / "runs"),
            seed=0,
        )
    )
    onnx_path, _manifest_path, manifest = export(
        ExportConfig(
            kind="classifier",
            checkpoint_path=str(ckpt),
            output_path=str(tmp_path / "clf.onnx"),
        )
    )
    assert manifest.output_shape == (1, 1)
    assert manifest.quantization == "fp32"
    tensor = torch.zeros((4, 256, 256), dtype=torch.float32)
    scenes = [GoldenClassifierScene(input_tensor=tensor, label_positive=True)]
    report = accept_classifier_artifact(
        str(onnx_path),
        manifest,
        scenes,
        run_inference=lambda t: 1.0,
        expected_input=(1, 4, 256, 256),
        expected_output=(1, 1),
        min_accuracy=0.9,
        max_latency_ms=10_000.0,
    )
    assert report.accepted
    promote(str(onnx_path), str(tmp_path / "active_classifier.onnx"), report)


@_skip_no_onnx
def test_export_int8_requires_onnxruntime(tmp_path: Path) -> None:
    """INT8 export raises ImportError when onnxruntime is missing."""
    if _HAS_ORT:
        pytest.skip("onnxruntime installed")
    ckpt = tmp_path / "seg.pt"
    train(
        TrainConfig(
            kind="segmentor",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=32,
            input_width_px=32,
            checkpoint_path=str(ckpt),
            run_dir=str(tmp_path / "runs"),
            seed=0,
        )
    )
    with pytest.raises(ImportError, match="onnxruntime"):
        export(
            ExportConfig(
                kind="segmentor",
                checkpoint_path=str(ckpt),
                output_path=str(tmp_path / "seg.onnx"),
                int8=True,
            )
        )


@_skip_no_onnx
@_skip_no_ort
def test_export_int8_writes_qdq_sibling(tmp_path: Path) -> None:
    """INT8 PTQ writes a sibling ONNX whose I/O stays float32."""
    ckpt = tmp_path / "seg.pt"
    pack_dir = tmp_path / "pack"
    images, masks, labels = make_synthetic_pack(6, 4, 32, 32, seed=0)
    write_processed_pack(pack_dir, images, masks, labels, SplitRecipe())
    train(
        TrainConfig(
            kind="segmentor",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=32,
            input_width_px=32,
            checkpoint_path=str(ckpt),
            run_dir=str(tmp_path / "runs"),
            seed=0,
        )
    )
    onnx_path, _manifest_path, manifest = export(
        ExportConfig(
            kind="segmentor",
            checkpoint_path=str(ckpt),
            output_path=str(tmp_path / "seg.onnx"),
            int8=True,
            calib_dir=str(pack_dir),
            calib_samples=2,
        )
    )
    assert manifest.quantization == "fp32"
    int8_path = int8_artifact_path(onnx_path)
    int8_manifest_path = int8_path.with_suffix(".json")
    assert int8_path.is_file()
    assert int8_manifest_path.is_file()
    int8_manifest = load_manifest(str(int8_manifest_path))
    assert int8_manifest.quantization == "int8"
    assert int8_manifest.input_shape == (1, 4, 32, 32)
    assert int8_manifest.output_shape == (1, 1, 32, 32)
    import onnxruntime as ort

    session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].type == "tensor(float)"
    assert session.get_outputs()[0].type == "tensor(float)"
    dummy = np.zeros((1, 4, 32, 32), dtype=np.float32)
    outputs = session.run(None, {session.get_inputs()[0].name: dummy})
    assert outputs[0].shape == (1, 1, 32, 32)


def test_promote_rejects_failed_gate(tmp_path: Path) -> None:
    """promote raises when the report is not accepted."""
    from tools.inference.accept import AcceptanceReport

    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"fake")
    report = AcceptanceReport(
        hash_ok=False,
        contract_ok=True,
        mean_iou=0.0,
        iou_ok=False,
        worst_latency_ms=0.0,
        latency_ok=True,
        accepted=False,
        detail="fail",
    )
    with pytest.raises(ValueError):
        promote(str(artifact), str(tmp_path / "dest.onnx"), report)
