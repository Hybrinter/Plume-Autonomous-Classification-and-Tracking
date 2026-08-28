"""ONNX export, manifest, and promote tests."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from tools.inference.accept import (
    GoldenClassifierScene,
    GoldenScene,
    accept_artifact,
    accept_classifier_artifact,
)
from tools.inference.export import ExportConfig, export, promote
from tools.inference.train import TrainConfig, train

_skip_no_onnx = pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None,
    reason="onnx extra not installed",
)


def _gold_mask_for(tensor: np.ndarray, scenes: list[GoldenScene]) -> np.ndarray:
    """Return the gold mask of the matching scene (perfect predictor)."""
    for scene in scenes:
        if scene.input_tensor is tensor:
            return scene.gold_mask
    return scenes[0].gold_mask


@_skip_no_onnx
def test_export_segmentor_then_accept(tmp_path: Path) -> None:
    """1-step synthetic 256 segmentor exports ONNX logits and passes injected accept."""
    ckpt = tmp_path / "seg.pt"
    train(
        TrainConfig(
            kind="segmentor",
            epochs=1,
            batch_size=2,
            synthetic_samples=2,
            checkpoint_path=str(ckpt),
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

    tensor = np.zeros((4, 256, 256), dtype=np.float32)
    gold = np.zeros((256, 256), dtype=np.float32)
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
            synthetic_samples=2,
            checkpoint_path=str(ckpt),
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
    tensor = np.zeros((4, 256, 256), dtype=np.float32)
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
