"""Model-acceptance gate tests (hash, I/O contract, golden-scene IoU, latency)."""

import hashlib
import json
from pathlib import Path

import numpy as np
from tools.accept import (
    GoldenScene,
    Manifest,
    accept_artifact,
    compute_iou,
    load_manifest,
)

_EXP_IN = (1, 4, 256, 256)
_EXP_OUT = (1, 1, 256, 256)


def _artifact(tmp_path: Path) -> tuple[str, Manifest]:
    """Write a fake artifact + a matching manifest; return (path, manifest)."""
    path = tmp_path / "model.onnx"
    path.write_bytes(b"fake-onnx-artifact")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = Manifest(
        version="v2",
        model_repo_sha="abc123",
        dataset_hash="ds456",
        input_shape=list(_EXP_IN),  # type: ignore[arg-type]
        output_shape=list(_EXP_OUT),  # type: ignore[arg-type]
        sha256=sha,
    )
    return str(path), manifest


def _scene(positive: bool) -> GoldenScene:
    """A golden scene whose gold mask has a positive region (or is empty)."""
    tensor = np.zeros((4, 8, 8), dtype=np.float32)
    gold = np.zeros((8, 8), dtype=np.float32)
    if positive:
        gold[2:6, 2:6] = 1.0
    return GoldenScene(input_tensor=tensor, gold_mask=gold)


def test_compute_iou() -> None:
    """compute_iou is 1.0 for identical masks and < 1 for partial overlap."""
    a = np.zeros((4, 4), dtype=np.float32)
    a[0:2, 0:2] = 1.0
    assert compute_iou(a, a) == 1.0
    b = np.zeros((4, 4), dtype=np.float32)
    b[1:3, 1:3] = 1.0
    assert 0.0 < compute_iou(a, b) < 1.0


def test_load_manifest_roundtrips(tmp_path: Path) -> None:
    """load_manifest parses a written manifest JSON."""
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": "v2",
                "model_repo_sha": "abc",
                "dataset_hash": "ds",
                "input_shape": [1, 4, 256, 256],
                "output_shape": [1, 1, 256, 256],
                "sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(str(path))
    assert manifest.version == "v2"
    assert manifest.input_shape == (1, 4, 256, 256)


def test_accept_passes_a_good_artifact(tmp_path: Path) -> None:
    """A correct hash + contract + perfect IoU + fast inference accepts the artifact."""
    path, manifest = _artifact(tmp_path)
    scenes = [_scene(positive=True), _scene(positive=False)]

    # Score each scene against its own gold (perfect predictor).
    report = accept_artifact(
        path,
        manifest,
        scenes,
        run_inference=lambda t: _gold_for(t, scenes),
        expected_input=_EXP_IN,
        expected_output=_EXP_OUT,
        min_iou=0.9,
        max_latency_ms=10_000.0,
    )
    assert report.hash_ok and report.contract_ok and report.iou_ok and report.latency_ok
    assert report.accepted


def test_accept_rejects_bad_hash(tmp_path: Path) -> None:
    """A manifest digest that does not match the artifact rejects the artifact."""
    path, manifest = _artifact(tmp_path)
    manifest = Manifest(
        version=manifest.version,
        model_repo_sha=manifest.model_repo_sha,
        dataset_hash=manifest.dataset_hash,
        input_shape=manifest.input_shape,
        output_shape=manifest.output_shape,
        sha256="0" * 64,
    )
    report = accept_artifact(
        path,
        manifest,
        [_scene(positive=True)],
        run_inference=lambda t: _gold_for(t, [_scene(positive=True)]),
        expected_input=_EXP_IN,
        expected_output=_EXP_OUT,
        min_iou=0.0,
        max_latency_ms=10_000.0,
    )
    assert not report.hash_ok
    assert not report.accepted


def test_accept_rejects_low_iou(tmp_path: Path) -> None:
    """A predictor that misses the plume fails the IoU gate."""
    path, manifest = _artifact(tmp_path)
    scenes = [_scene(positive=True)]
    report = accept_artifact(
        path,
        manifest,
        scenes,
        run_inference=lambda t: np.zeros((8, 8), dtype=np.float32),  # predicts nothing
        expected_input=_EXP_IN,
        expected_output=_EXP_OUT,
        min_iou=0.5,
        max_latency_ms=10_000.0,
    )
    assert not report.iou_ok
    assert not report.accepted


def _gold_for(tensor: np.ndarray, scenes: list[GoldenScene]) -> np.ndarray:
    """Return the gold mask of the scene whose input tensor matches (perfect predictor stub)."""
    for scene in scenes:
        if scene.input_tensor is tensor:
            return scene.gold_mask
    return scenes[0].gold_mask
