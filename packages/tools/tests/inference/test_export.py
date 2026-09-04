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
    convert_fp16,
    export,
    fp16_artifact_path,
    int8_artifact_path,
    promote,
    quantize_knee,
    reexport_spatial,
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


def test_fp16_artifact_path() -> None:
    """FP16 sibling uses the FP32 stem plus `.fp16.onnx`."""
    assert fp16_artifact_path("/tmp/cls.onnx") == Path("/tmp/cls.fp16.onnx")


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
    assert batches[0].dtype == torch.float32
    assert float(batches[0].min().item()) >= 0.0
    assert float(batches[0].max().item()) <= 1.0


def test_calibration_batches_from_pack(tmp_path: Path) -> None:
    """A processed pack supplies train-split calibration tensors."""
    from tools.inference.export import _calibration_batches

    images, masks, labels = make_synthetic_pack(6, 4, 8, 8, seed=0)
    write_processed_pack(tmp_path, images, masks, labels, SplitRecipe())
    batches = _calibration_batches((1, 4, 8, 8), str(tmp_path), 2)
    assert len(batches) == 2
    assert batches[0].shape == (1, 4, 8, 8)
    assert batches[0].dtype == torch.float32


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
def test_export_dilatenet_resizes_logits_to_input_hw(tmp_path: Path) -> None:
    """A decoder-free dilatenet exports full-resolution logits through bilinear resize."""
    ckpt = tmp_path / "dilate.pt"
    train(
        TrainConfig(
            kind="segmentor",
            arch="dilatenet_w32",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=32,
            input_width_px=32,
            checkpoint_path=str(ckpt),
            run_dir=str(tmp_path / "runs"),
            seed=0,
            device="cpu",
        )
    )
    onnx_path, _manifest_path, manifest = export(
        ExportConfig(
            kind="segmentor",
            checkpoint_path=str(ckpt),
            output_path=str(tmp_path / "dilate.onnx"),
            input_height_px=32,
            input_width_px=32,
        )
    )
    assert onnx_path.is_file()
    assert manifest.output_shape == (1, 1, 32, 32)
    if importlib.util.find_spec("onnxruntime") is not None:
        from tools.inference.accept import onnx_inference_fn

        tensor = torch.zeros((4, 32, 32), dtype=torch.float32)
        pred = onnx_inference_fn(str(onnx_path))(tensor)
        assert pred.shape == (32, 32)


@_skip_no_onnx
def test_export_override_spatial_uses_config_hw(tmp_path: Path) -> None:
    """``override_spatial`` exports at ExportConfig H/W, not the checkpoint size."""
    ckpt = tmp_path / "dilate.pt"
    train(
        TrainConfig(
            kind="segmentor",
            arch="dilatenet_w32",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=32,
            input_width_px=32,
            checkpoint_path=str(ckpt),
            run_dir=str(tmp_path / "runs"),
            seed=0,
            device="cpu",
        )
    )
    _onnx_path, _manifest_path, manifest = export(
        ExportConfig(
            kind="segmentor",
            checkpoint_path=str(ckpt),
            output_path=str(tmp_path / "dilate.onnx"),
            input_height_px=48,
            input_width_px=64,
            override_spatial=True,
        )
    )
    assert manifest.input_shape == (1, 4, 48, 64)
    assert manifest.output_shape == (1, 1, 48, 64)


@_skip_no_onnx
def test_reexport_spatial_copies_weights_and_changes_hw(tmp_path: Path) -> None:
    """Spatial re-export keeps trained weights and writes a new I/O contract."""
    ckpt = tmp_path / "dilate.pt"
    train(
        TrainConfig(
            kind="segmentor",
            arch="dilatenet_w32",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=16,
            input_width_px=16,
            checkpoint_path=str(ckpt),
            run_dir=str(tmp_path / "runs"),
            seed=0,
            device="cpu",
        )
    )
    src, _src_manifest_path, src_manifest = export(
        ExportConfig(
            kind="segmentor",
            checkpoint_path=str(ckpt),
            output_path=str(tmp_path / "dilate16.onnx"),
            input_height_px=16,
            input_width_px=16,
        )
    )
    dest, dest_manifest_path, dest_manifest = reexport_spatial(
        str(src),
        str(tmp_path / "dilate32.onnx"),
        kind="segmentor",
        arch="dilatenet_w32",
        height=32,
        width=32,
    )
    assert dest.is_file()
    assert dest_manifest_path.is_file()
    assert dest_manifest.input_shape == (1, 4, 32, 32)
    assert dest_manifest.output_shape == (1, 1, 32, 32)
    assert dest_manifest.dataset_hash == src_manifest.dataset_hash
    assert dest_manifest.sha256 != src_manifest.sha256
    if importlib.util.find_spec("onnxruntime") is not None:
        from tools.inference.accept import onnx_inference_fn

        tensor = torch.zeros((4, 32, 32), dtype=torch.float32)
        pred = onnx_inference_fn(str(dest))(tensor)
        assert pred.shape == (32, 32)


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


@_skip_no_onnx
@_skip_no_ort
def test_convert_fp16_keeps_float_io(tmp_path: Path) -> None:
    """FP16 conversion keeps graph I/O as float32 and records quantization."""
    ckpt = tmp_path / "cls.pt"
    train(
        TrainConfig(
            kind="classifier",
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
            kind="classifier",
            checkpoint_path=str(ckpt),
            output_path=str(tmp_path / "cls.onnx"),
        )
    )
    assert manifest.quantization == "fp32"
    dest = tmp_path / "cls.fp16.onnx"
    out_path, out_json, fp16_manifest = convert_fp16(str(onnx_path), str(dest))
    assert out_path == dest
    assert out_json == dest.with_suffix(".json")
    assert fp16_manifest.quantization == "fp16"
    assert fp16_manifest.input_shape == (1, 4, 32, 32)
    assert fp16_manifest.output_shape == (1, 1)
    import onnxruntime as ort

    session = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].type == "tensor(float)"
    assert session.get_outputs()[0].type == "tensor(float)"
    dummy = np.zeros((1, 4, 32, 32), dtype=np.float32)
    outputs = session.run(None, {session.get_inputs()[0].name: dummy})
    assert outputs[0].shape == (1, 1)


@_skip_no_onnx
@_skip_no_ort
def test_quantize_knee_overwrites_with_mixed_precision(tmp_path: Path) -> None:
    """Knee conversion writes classifier FP16 and segmentor INT8 in place."""
    cls_ckpt = tmp_path / "cls.pt"
    seg_ckpt = tmp_path / "seg.pt"
    train(
        TrainConfig(
            kind="classifier",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=32,
            input_width_px=32,
            checkpoint_path=str(cls_ckpt),
            run_dir=str(tmp_path / "runs_cls"),
            seed=0,
        )
    )
    train(
        TrainConfig(
            kind="segmentor",
            epochs=1,
            batch_size=2,
            synthetic_samples=4,
            input_height_px=32,
            input_width_px=32,
            checkpoint_path=str(seg_ckpt),
            run_dir=str(tmp_path / "runs_seg"),
            seed=0,
        )
    )
    cls_onnx, _, _ = export(
        ExportConfig(
            kind="classifier",
            checkpoint_path=str(cls_ckpt),
            output_path=str(tmp_path / "active_classifier.onnx"),
        )
    )
    seg_onnx, _, _ = export(
        ExportConfig(
            kind="segmentor",
            checkpoint_path=str(seg_ckpt),
            output_path=str(tmp_path / "active_segmentor.onnx"),
        )
    )
    (cls_out, _, cls_manifest), (seg_out, _, seg_manifest) = quantize_knee(
        str(cls_onnx),
        str(seg_onnx),
        calib_samples=2,
    )
    assert cls_out == cls_onnx
    assert seg_out == seg_onnx
    assert cls_manifest.quantization == "fp16"
    assert seg_manifest.quantization == "int8"
    assert load_manifest(str(cls_onnx.with_suffix(".json"))).quantization == "fp16"
    assert load_manifest(str(seg_onnx.with_suffix(".json"))).quantization == "int8"
    import onnxruntime as ort

    dummy = np.zeros((1, 4, 32, 32), dtype=np.float32)
    cls_session = ort.InferenceSession(str(cls_onnx), providers=["CPUExecutionProvider"])
    seg_session = ort.InferenceSession(str(seg_onnx), providers=["CPUExecutionProvider"])
    assert cls_session.get_inputs()[0].type == "tensor(float)"
    assert seg_session.get_inputs()[0].type == "tensor(float)"
    cls_out_arr = cls_session.run(None, {cls_session.get_inputs()[0].name: dummy})[0]
    seg_out_arr = seg_session.run(None, {seg_session.get_inputs()[0].name: dummy})[0]
    assert cls_out_arr.shape == (1, 1)
    assert seg_out_arr.shape == (1, 1, 32, 32)


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
