"""Model-acceptance gate tests (hash, I/O contract, golden-scene IoU, latency)."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from tools.inference.accept import (
    GoldenClassifierScene,
    GoldenScene,
    Manifest,
    _latency_stats,
    accept_artifact,
    accept_classifier_artifact,
    compute_iou,
    load_golden_classifier_scenes,
    load_golden_scenes,
    load_manifest,
)
from tools.inference.data import load_processed_pack, make_synthetic_pack, write_processed_pack
from tools.inference.split import SplitRecipe

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
    tensor = torch.zeros((4, 8, 8), dtype=torch.float32)
    gold = torch.zeros((8, 8), dtype=torch.float32)
    if positive:
        gold[2:6, 2:6] = 1.0
    return GoldenScene(input_tensor=tensor, gold_mask=gold)


def test_compute_iou() -> None:
    """compute_iou is 1.0 for identical masks and < 1 for partial overlap."""
    a = torch.zeros((4, 4), dtype=torch.float32)
    a[0:2, 0:2] = 1.0
    assert compute_iou(a, a) == 1.0
    b = torch.zeros((4, 4), dtype=torch.float32)
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
    assert manifest.quantization == "fp32"


def test_load_manifest_reads_quantization(tmp_path: Path) -> None:
    """load_manifest reads an explicit quantization field."""
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
                "quantization": "int8",
            }
        ),
        encoding="utf-8",
    )
    assert load_manifest(str(path)).quantization == "int8"


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
        run_inference=lambda t: torch.zeros((8, 8), dtype=torch.float32),  # predicts nothing
        expected_input=_EXP_IN,
        expected_output=_EXP_OUT,
        min_iou=0.5,
        max_latency_ms=10_000.0,
    )
    assert not report.iou_ok
    assert not report.accepted


def _gold_for(tensor: torch.Tensor, scenes: list[GoldenScene]) -> torch.Tensor:
    """Return the gold mask of the scene whose input tensor matches (perfect predictor stub)."""
    for scene in scenes:
        if scene.input_tensor is tensor:
            return scene.gold_mask
    return scenes[0].gold_mask


def _write_pack(tmp_path: Path, n: int = 6, seed: int = 0) -> Path:
    """Write a synthetic processed pack and return its directory."""
    images, masks, labels = make_synthetic_pack(n, 4, 8, 8, seed=seed)
    write_processed_pack(
        tmp_path,
        images=images,
        masks=masks,
        labels=labels,
        recipe=SplitRecipe(seed=seed),
        source_doi="synthetic",
    )
    return tmp_path


def test_load_golden_scenes_shapes_and_count(tmp_path: Path) -> None:
    """load_golden_scenes returns one test-split scene per sample with correct shapes."""
    pack_dir = _write_pack(tmp_path)
    pack = load_processed_pack(pack_dir)
    scenes = load_golden_scenes(str(pack_dir), split="test")
    assert len(scenes) == len(pack.splits.test)
    for scene, index in zip(scenes, pack.splits.test, strict=True):
        assert scene.input_tensor.shape == (4, 8, 8)
        assert scene.gold_mask.shape == (8, 8)
        assert torch.equal(scene.input_tensor, pack.images[index])
        assert torch.equal(scene.gold_mask, pack.masks[index, 0])


def test_load_golden_classifier_scenes_labels(tmp_path: Path) -> None:
    """load_golden_classifier_scenes maps pack labels to label_positive booleans."""
    pack_dir = _write_pack(tmp_path)
    pack = load_processed_pack(pack_dir)
    scenes = load_golden_classifier_scenes(str(pack_dir), split="test")
    assert len(scenes) == len(pack.splits.test)
    for scene, index in zip(scenes, pack.splits.test, strict=True):
        assert scene.label_positive == bool(float(pack.labels[index, 0]) >= 0.5)


def test_load_golden_scenes_limit(tmp_path: Path) -> None:
    """limit truncates the scene list; zero returns the whole split."""
    pack_dir = _write_pack(tmp_path)
    pack = load_processed_pack(pack_dir)
    full = load_golden_scenes(str(pack_dir), split="test", limit=0)
    assert len(full) == len(pack.splits.test)
    limited = load_golden_scenes(str(pack_dir), split="test", limit=1)
    assert len(limited) == 1
    assert torch.equal(limited[0].input_tensor, full[0].input_tensor)
    assert torch.equal(limited[0].gold_mask, full[0].gold_mask)


def test_load_golden_scenes_rejects_unknown_split(tmp_path: Path) -> None:
    """An unknown split name raises ValueError."""
    pack_dir = _write_pack(tmp_path)
    with pytest.raises(ValueError, match="unknown split"):
        load_golden_scenes(str(pack_dir), split="holdout")


def test_loaded_scenes_pass_iou_gate(tmp_path: Path) -> None:
    """Pack-loaded golden scenes let accept_artifact report iou_ok when predictions match."""
    pack_dir = _write_pack(tmp_path)
    scenes = load_golden_scenes(str(pack_dir), split="test")
    path, manifest = _artifact(tmp_path)
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
    assert report.iou_ok
    assert report.mean_iou == 1.0


def test_latency_stats_empty_median_and_p95() -> None:
    """An empty sample list reports zeros. Twenty samples report worst, median, p95."""
    assert _latency_stats([]) == (0.0, 0.0, 0.0)
    samples = [float(n) for n in range(1, 21)]
    worst, median, p95 = _latency_stats(samples)
    assert worst == 20.0
    assert median == 10.5
    assert p95 == 19.0


def test_accept_classifier_passes_with_injected_logits(tmp_path: Path) -> None:
    """Classifier gate accepts matching hash, contract, accuracy, and latency."""
    path, manifest = _artifact(tmp_path)
    manifest = Manifest(
        version=manifest.version,
        model_repo_sha=manifest.model_repo_sha,
        dataset_hash=manifest.dataset_hash,
        input_shape=(1, 4, 256, 256),
        output_shape=(1, 1),
        sha256=manifest.sha256,
    )
    tensor_pos = torch.ones((4, 8, 8), dtype=torch.float32)
    tensor_neg = torch.zeros((4, 8, 8), dtype=torch.float32)
    scenes = [
        GoldenClassifierScene(input_tensor=tensor_pos, label_positive=True),
        GoldenClassifierScene(input_tensor=tensor_neg, label_positive=False),
    ]

    def _run(tensor: torch.Tensor) -> float:
        return 1.0 if tensor is tensor_pos else -1.0

    report = accept_classifier_artifact(
        path,
        manifest,
        scenes,
        run_inference=_run,
        expected_input=(1, 4, 256, 256),
        expected_output=(1, 1),
        min_accuracy=1.0,
        max_latency_ms=10_000.0,
    )
    assert report.accuracy_ok and report.hash_ok and report.contract_ok
    assert report.accepted


def test_accept_classifier_rejects_low_accuracy(tmp_path: Path) -> None:
    """A predictor that always reports negative fails the accuracy gate."""
    path, manifest = _artifact(tmp_path)
    manifest = Manifest(
        version=manifest.version,
        model_repo_sha=manifest.model_repo_sha,
        dataset_hash=manifest.dataset_hash,
        input_shape=(1, 4, 256, 256),
        output_shape=(1, 1),
        sha256=manifest.sha256,
    )
    scenes = [
        GoldenClassifierScene(
            input_tensor=torch.zeros((4, 8, 8), dtype=torch.float32),
            label_positive=True,
        )
    ]
    report = accept_classifier_artifact(
        path,
        manifest,
        scenes,
        run_inference=lambda t: -4.0,
        expected_input=(1, 4, 256, 256),
        expected_output=(1, 1),
        min_accuracy=0.9,
        max_latency_ms=10_000.0,
    )
    assert not report.accuracy_ok
    assert not report.accepted
