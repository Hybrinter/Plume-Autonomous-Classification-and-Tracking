"""Real-SDK export parity tests.

Every test in this module requires the optional onnx and onnxruntime
packages. When they are absent the module reports skips; skips are not parity
passes.
"""

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from flight.libs.config import InferenceConfig, PactConfig
from flight.libs.types import FaultCode, Ok, Result
from flight.payload.gimbal.footprint import GSD_REFERENCE_M, to_model_gsd
from flight.payload.inference.onnx_session import load_onnx_session
from tools.ml_models.arch.film import CONDITIONING_ID, GsdFilm
from tools.ml_models.arch.registry import build
from tools.ml_models.export.export import ExportConfig, export
from tools.ml_models.export.manifest import ModelManifest, load_manifest, sidecar_path
from tools.ml_models.export.session import Session, open_session

onnx = pytest.importorskip("onnx", reason="onnx SDK not installed")
ort = pytest.importorskip("onnxruntime", reason="onnxruntime SDK not installed")

FULL_COVERAGE = ((10.0, 10.0), (45.0, 45.0))


def _checkpoint(
    tmp_path: Path,
    kind: str = "classifier",
    arch: str = "pactnet",
    *,
    perturb_film: bool = False,
    seed: int = 0,
) -> tuple[Path, torch.nn.Module]:
    """Build a model and write a training-format checkpoint."""
    torch.manual_seed(seed)
    model = build(kind, arch, 3)
    if perturb_film:
        with torch.no_grad():
            for name in ("stem_film", "head_film"):
                film = cast(GsdFilm, getattr(model, name))
                last = cast(torch.nn.Linear, film.net[-1])
                last.weight.normal_(0.0, 0.05)
                last.bias.normal_(0.0, 0.05)
    model.eval()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "kind": kind,
            "arch": arch,
            "state_dict": model.state_dict(),
            "epoch": 0,
            "conditioning": CONDITIONING_ID,
            "config": {},
            "provenance": {
                "in_channels": 3,
                "band_names": ["BLUE", "GREEN", "RED"],
                "norm": "unit",
                "gsd_reference_m": GSD_REFERENCE_M,
                "gsd_min_m": list(FULL_COVERAGE[0]),
                "gsd_max_m": list(FULL_COVERAGE[1]),
                "train_samples": 8,
            },
            "dataset_weights": [1.0],
            "dataset_hash": hashlib.sha256(b"dataset").hexdigest(),
            "input_height_px": 193,
            "input_width_px": 258,
        },
        path,
    )
    return path, model


def _encoded_gsd(raw: tuple[float, float], batch: int) -> torch.Tensor:
    """Encode a raw (lateral, along) GSD pair for ``batch`` rows."""
    encoded = to_model_gsd(np.asarray([raw]), GSD_REFERENCE_M)
    assert isinstance(encoded, Ok)
    row = torch.from_numpy(np.asarray(encoded.value, dtype=np.float32))
    return row.expand(batch, 2).contiguous()


def _ort_output(session: Session, image: torch.Tensor, gsd: torch.Tensor) -> np.ndarray:
    outputs = session.run(
        None,
        {
            "image": image.numpy().astype(np.float32),
            "gsd": gsd.numpy().astype(np.float32),
        },
    )
    return np.asarray(outputs[0])


@pytest.mark.parametrize("batch", [1, 3, 64])
def test_classifier_export_parity(tmp_path: Path, batch: int) -> None:
    """Torch and ORT agree on exported classifier logits across batch sizes."""
    ckpt, model = _checkpoint(tmp_path)
    artifact = tmp_path / "model.onnx"
    result = export(ExportConfig(str(ckpt), str(artifact)))
    assert isinstance(result, Ok), result
    manifest = load_manifest(sidecar_path(artifact))
    assert manifest.partial_gsd is False
    session_result = open_session(artifact, manifest)
    assert isinstance(session_result, Ok), session_result
    session = session_result.value

    torch.manual_seed(7)
    image = torch.randn(batch, 3, 193, 258)
    gsd = _encoded_gsd((24.0, 38.7), batch)
    with torch.no_grad():
        expected = model(image, gsd).numpy()
    actual = _ort_output(session, image, gsd)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("height,width", [(120, 120), (31, 50), (193, 258)])
def test_dynamic_spatial_parity(tmp_path: Path, height: int, width: int) -> None:
    """Dynamic spatial dims accept varied tile sizes."""
    ckpt, model = _checkpoint(tmp_path)
    artifact = tmp_path / "model.onnx"
    result = export(ExportConfig(str(ckpt), str(artifact), dynamic_spatial=True))
    assert isinstance(result, Ok), result
    manifest = load_manifest(sidecar_path(artifact))
    session_result = open_session(artifact, manifest)
    assert isinstance(session_result, Ok), session_result

    torch.manual_seed(3)
    image = torch.randn(2, 3, height, width)
    gsd = _encoded_gsd((16.0, 16.0), 2)
    with torch.no_grad():
        expected = model(image, gsd).numpy()
    actual = _ort_output(session_result.value, image, gsd)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("batch", [1, 3, 64])
def test_segmentor_export_parity(tmp_path: Path, batch: int) -> None:
    """Torch and ORT agree on exported segmentor logits across batch sizes."""
    ckpt, model = _checkpoint(tmp_path, "segmentor", "dilatenet_w16", perturb_film=True)
    artifact = tmp_path / "seg.onnx"
    result = export(ExportConfig(str(ckpt), str(artifact)))
    assert isinstance(result, Ok), result
    manifest = load_manifest(sidecar_path(artifact))
    assert manifest.kind == "segmentor"
    session_result = open_session(artifact, manifest)
    assert isinstance(session_result, Ok), session_result

    torch.manual_seed(5)
    image = torch.randn(batch, 3, 193, 258)
    gsd = _encoded_gsd((24.0, 38.7), batch)
    with torch.no_grad():
        expected = model(image, gsd).numpy()
    actual = _ort_output(session_result.value, image, gsd)
    assert actual.shape == (batch, 1, 193, 258)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("height,width", [(120, 120), (31, 50), (193, 258)])
def test_segmentor_dynamic_spatial_parity(tmp_path: Path, height: int, width: int) -> None:
    """Dynamic spatial dims hold for the segmentor with a nonzero FiLM."""
    ckpt, model = _checkpoint(tmp_path, "segmentor", "dilatenet_w16", perturb_film=True)
    artifact = tmp_path / "seg.onnx"
    result = export(ExportConfig(str(ckpt), str(artifact), dynamic_spatial=True))
    assert isinstance(result, Ok), result
    manifest = load_manifest(sidecar_path(artifact))
    session_result = open_session(artifact, manifest)
    assert isinstance(session_result, Ok), session_result

    torch.manual_seed(9)
    image = torch.randn(3, 3, height, width)
    gsd = _encoded_gsd((16.0, 16.0), 3)
    with torch.no_grad():
        expected = model(image, gsd).numpy()
    actual = _ort_output(session_result.value, image, gsd)
    assert actual.shape == (3, 1, height, width)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_gsd_changes_outputs(tmp_path: Path) -> None:
    """With a non-identity FiLM, encoded GSD shifts both backends' outputs."""
    ckpt, model = _checkpoint(tmp_path, perturb_film=True)
    artifact = tmp_path / "model.onnx"
    result = export(ExportConfig(str(ckpt), str(artifact)))
    assert isinstance(result, Ok), result
    manifest = load_manifest(sidecar_path(artifact))
    session_result = open_session(artifact, manifest)
    assert isinstance(session_result, Ok), session_result

    torch.manual_seed(11)
    image = torch.randn(2, 3, 193, 258)
    near = _encoded_gsd((16.0, 16.0), 2)
    far = _encoded_gsd((24.0, 38.7), 2)
    with torch.no_grad():
        torch_near = model(image, near).numpy()
        torch_far = model(image, far).numpy()
    ort_near = _ort_output(session_result.value, image, near)
    ort_far = _ort_output(session_result.value, image, far)
    assert not np.allclose(torch_near, torch_far)
    assert not np.allclose(ort_near, ort_far)
    np.testing.assert_allclose(ort_near, torch_near, atol=1e-4, rtol=1e-4)


def test_export_manifest_contents(tmp_path: Path) -> None:
    """The sidecar records graph-declared shapes and training coverage."""
    ckpt, _model = _checkpoint(tmp_path)
    artifact = tmp_path / "model.onnx"
    result = export(ExportConfig(str(ckpt), str(artifact)))
    assert isinstance(result, Ok), result
    manifest = load_manifest(sidecar_path(artifact))
    assert manifest.sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert manifest.input_shape == (None, 3, 193, 258)
    assert manifest.gsd_input_shape == (None, 2)
    assert manifest.output_shape == (None, 1)
    assert manifest.gsd_min_m == FULL_COVERAGE[0]
    assert manifest.gsd_max_m == FULL_COVERAGE[1]
    assert manifest.gsd_reference_m == GSD_REFERENCE_M
    raw = json.loads(sidecar_path(artifact).read_text())
    assert raw["schema"] == 2
    assert raw["partial_gsd"] is False


def test_exported_pair_loads_with_flight_metadata_and_config(tmp_path: Path) -> None:
    """A generated fixed-tile pair loads and stages with its configured metadata."""
    import json

    from flight.core.model_deploy import ModelDeployService, parse_manifest
    from flight.libs.bus import MessageBus
    from flight.libs.messages import ModelStagedMsg, RoutedCommandMsg
    from flight.libs.time import ManualClock
    from flight.libs.types import MessageType, ModelDeployState
    from tools.ml_models.dataset.build import build_synthetic
    from tools.ml_models.export.accept import accept_artifact
    from tools.ml_models.export.manifest import acceptance_path
    from tools.ml_models.export.pair import write_pair_manifest

    inference = InferenceConfig()
    artifacts: dict[str, Path] = {}
    manifests: dict[str, ModelManifest] = {}
    dataset = tmp_path / "synthetic"
    build_synthetic(dataset, n=9)
    for kind, arch, filename in (
        ("classifier", "pactnet", "classifier.onnx"),
        ("segmentor", "dilatenet_w16", "segmentor.onnx"),
    ):
        checkpoint, _model = _checkpoint(tmp_path, kind, arch)
        artifact = tmp_path / filename
        result = export(ExportConfig(str(checkpoint), str(artifact), dynamic_spatial=False))
        assert isinstance(result, Ok), result
        manifest = load_manifest(sidecar_path(artifact))
        tile_hw = (
            inference.input_height_px // manifest.grid[0],
            inference.input_width_px // manifest.grid[1],
        )
        assert manifest.tile_hw == tile_hw == (193, 258)
        assert manifest.gsd_reference_m == inference.gsd_reference_m
        expected_input = (None, len(inference.input_bands), *tile_hw)
        expected_output = (None, 1) if kind == "classifier" else (None, 1, *tile_hw)
        session = load_onnx_session(
            str(artifact),
            expected_sha256=manifest.sha256,
            expected_input_shape=expected_input,
            expected_output_shape=expected_output,
            expected_gsd_shape=(None, 2),
        )
        assert set(node.name for node in session.get_inputs()) == {"image", "gsd"}
        acceptance = accept_artifact(
            artifact,
            manifest,
            [dataset],
            min_iou=0.0,
            min_accuracy=0.0,
            max_latency_ms=1000.0,
        )
        assert acceptance["accepted"] is True
        acceptance_path(artifact).write_text(
            json.dumps({"accepted": True, "sha256": manifest.sha256}), encoding="utf-8"
        )
        artifacts[kind] = artifact
        manifests[kind] = manifest
    assert (
        manifests["classifier"].band_names
        == manifests["segmentor"].band_names
        == inference.input_bands
    )
    assert manifests["classifier"].gsd_reference_m == manifests["segmentor"].gsd_reference_m

    pair_path = tmp_path / "pair.json"
    pair_result = write_pair_manifest(
        sidecar_path(artifacts["classifier"]),
        sidecar_path(artifacts["segmentor"]),
        pair_path,
    )
    assert isinstance(pair_result, Ok), pair_result
    pair_blob = pair_path.read_bytes()
    parsed = parse_manifest(pair_blob)
    assert parsed is not None
    assert parsed.grid == (inference.tile_rows, inference.tile_cols)
    assert parsed.frame_hw == (inference.input_height_px, inference.input_width_px)
    assert parsed.tile_hw == (
        inference.input_height_px // inference.tile_rows,
        inference.input_width_px // inference.tile_cols,
    )

    class _Storage:
        def read(self, _entry_id: str) -> Result[bytes, FaultCode]:
            return Ok(pair_blob)

    bus = MessageBus()
    service = ModelDeployService.from_config(
        PactConfig(),
        bus,
        ManualClock(),
        _Storage(),
    )
    bus.publish(
        ModelStagedMsg(
            msg_type=MessageType.MODEL_STAGED,
            timestamp_utc="t",
            entry_id="pair-entry",
            sha256=hashlib.sha256(pair_blob).hexdigest(),
            version="",
        )
    )
    service.tick()
    assert service.state.state is ModelDeployState.STAGED
    bus.publish(
        RoutedCommandMsg(
            msg_type=MessageType.ROUTED_COMMAND,
            timestamp_utc="t",
            target="model_deploy",
            command_id="ACTIVATE_MODEL",
            params={"version": parsed.version},
            source="ground",
            seq=1,
        )
    )
    service.tick()
    assert service.state.state.name == ModelDeployState.ACTIVE.name
    assert service.state.active_version == parsed.version


def test_dynamic_spatial_research_export_fails_flight_shape_gate(tmp_path: Path) -> None:
    """A valid dynamic research graph does not satisfy configured flight tiles."""
    inference = InferenceConfig()
    checkpoint, _model = _checkpoint(tmp_path)
    artifact = tmp_path / "dynamic.onnx"
    result = export(ExportConfig(str(checkpoint), str(artifact), dynamic_spatial=True))
    assert isinstance(result, Ok), result
    manifest = load_manifest(sidecar_path(artifact))
    with pytest.raises(ValueError, match="model I/O contract verification failed"):
        load_onnx_session(
            str(artifact),
            expected_sha256=manifest.sha256,
            expected_input_shape=(None, len(inference.input_bands), 193, 258),
            expected_output_shape=(None, 1),
            expected_gsd_shape=(None, 2),
        )


def test_export_refuses_existing_files(tmp_path: Path) -> None:
    """Existing artifact or sidecar paths are never overwritten."""
    from flight.libs.types import Err

    ckpt, _model = _checkpoint(tmp_path)
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"keep")
    result = export(ExportConfig(str(ckpt), str(artifact)))
    assert isinstance(result, Err)
    assert artifact.read_bytes() == b"keep"


def test_full_acceptance_flow(tmp_path: Path) -> None:
    """accept_artifact scores a real session on a finished test split."""
    from tools.ml_models.dataset.build import build_synthetic
    from tools.ml_models.export.accept import accept_artifact

    ckpt, _model = _checkpoint(tmp_path)
    artifact = tmp_path / "model.onnx"
    result = export(ExportConfig(str(ckpt), str(artifact)))
    assert isinstance(result, Ok), result
    manifest = load_manifest(sidecar_path(artifact))
    dataset = tmp_path / "ds"
    build_synthetic(dataset, n=9)
    report = accept_artifact(
        artifact,
        manifest,
        [dataset],
        min_iou=0.0,
        min_accuracy=0.0,
        max_latency_ms=1000.0,
    )
    assert report["hash_ok"] is True
    assert report["contract_ok"] is True
    assert report["accepted"] is True
    assert report["metric"] == "accuracy"
