"""Shared evidence acceptance with SDK fakes and explicit unavailable populations."""

from collections.abc import Callable
from dataclasses import replace
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.dataset.manifest import (
    ShardCount,
    compute_dataset_hash,
    load_manifest,
    write_manifest,
)
from tools.ml_models.dataset.store import RowRecord, ShardWriter
from tools.ml_models.export.accept import accept_artifact
from tools.ml_models.export.manifest import ModelManifest, acceptance_path
from tools.ml_models.export.session import Node


def _manifest(**overrides: object) -> ModelManifest:
    fields: dict[str, object] = {
        "version": "a" * 16,
        "kind": "classifier",
        "arch": "pactnet",
        "sha256": "0" * 64,
        "dataset_hash": "b" * 64,
        "band_names": ("BLUE", "GREEN", "RED"),
        "input_shape": (None, 3, None, None),
        "gsd_input_shape": (None, 2),
        "output_shape": (None, 1),
        "gsd_reference_m": 15.87,
        "gsd_min_m": (14.0, 14.0),
        "gsd_max_m": (40.0, 40.0),
        "conditioning": "film-log-gsd-v1",
        "gsd_encoding": "ln_metres_over_reference_lateral_along",
    }
    fields.update(overrides)
    return ModelManifest(**fields)  # type: ignore[arg-type]


def test_accept_rejects_invalid_thresholds(tmp_path: Path) -> None:
    """Out-of-range thresholds fail before any dataset or model loads."""
    for kwargs in (
        {"min_iou": -0.1},
        {"min_iou": 1.1},
        {"min_accuracy": float("nan")},
        {"max_latency_ms": 0.0},
    ):
        result = accept_artifact(tmp_path / "m.onnx", _manifest(), tmp_path, **kwargs)
        assert isinstance(result, Err)
        assert "threshold" in result.error


def test_accept_rejects_reference_mismatch(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A dataset GSD reference that disagrees with the model is rejected."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=6)
    result = accept_artifact(
        tmp_path / "m.onnx",
        _manifest(gsd_reference_m=16.0),
        dataset,
    )
    assert isinstance(result, Err)
    assert "disagree" in result.error


def test_accept_rejects_band_mismatch(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A dataset band order that disagrees with the model is rejected."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=6)
    manifest = _manifest()
    object.__setattr__(manifest, "band_names", ("A", "B", "C"))
    result = accept_artifact(
        tmp_path / "m.onnx",
        manifest,
        dataset,
    )
    assert isinstance(result, Err)
    assert "disagree" in result.error


def test_accept_rejects_empty_dataset(tmp_path: Path) -> None:
    """An empty dataset path is rejected before any loading."""
    result = accept_artifact(tmp_path / "m.onnx", _manifest(), "")
    assert isinstance(result, Err)
    assert "exactly one finished dataset" in result.error


def test_accept_rejects_fixed_input_shape_mismatch(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """Fixed model H/W mismatch fails before the unavailable scorer is reached."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=6)
    artifact = tmp_path / "m.onnx"
    artifact.write_bytes(b"stub")
    for shape in ((None, 3, 120, 258), (None, 3, 193, 120), (None, 3, 9, 9)):
        manifest = _manifest()
        object.__setattr__(manifest, "input_shape", shape)
        result = accept_artifact(artifact, manifest, dataset)
        assert isinstance(result, Err)
        assert "input_shape" in result.error


def test_accept_shared_test_evidence(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A perfect SDK fake uses every test row and real shared metric evidence."""
    import tools.ml_models.export.accept as accept

    dataset = build_synthetic_dataset(tmp_path / "ds", n=6)
    artifact = tmp_path / "m.onnx"
    artifact.write_bytes(b"stub")
    session = _Session()
    monkeypatch.setattr(accept, "open_session", lambda *_args: Ok(session))
    result = accept_artifact(artifact, _manifest(), dataset, min_accuracy=1.0, max_latency_ms=1e9)
    assert isinstance(result, Ok), result
    assert result.value["accepted"] is True
    assert result.value["metric"] == "accuracy"
    assert result.value["sha256"] == _manifest().sha256
    evaluation = result.value["evaluation"]
    assert isinstance(evaluation, dict)
    assert evaluation["split"] == "test"
    manifest = load_manifest(dataset / "dataset.json")
    assert session.n == sum(
        shard.n for shard in manifest.shards if shard.task == "classifier" and shard.split == "test"
    )
    metrics = evaluation["metrics"]
    assert isinstance(metrics, (list, tuple))
    assert next(metric["value"] for metric in metrics if metric["name"] == "accuracy") == 1.0
    assert not acceptance_path(artifact).exists()


class _Session:
    def __init__(self, invalid: bool = False, inverted: bool = False) -> None:
        self.n = 0
        self.invalid = invalid
        self.inverted = inverted

    def get_inputs(self) -> list[Node]:
        return []

    def get_outputs(self) -> list[Node]:
        return []

    def run(self, output_names: list[str] | None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        image = feeds["image"]
        assert image.shape[0] == 1
        assert feeds["gsd"].shape == (1, 2)
        self.n += 1
        logits = np.where(image.max(axis=(1, 2, 3)) > 0.5, 4.0, -4.0).astype(np.float32)
        if self.inverted:
            logits = -logits
        if self.invalid:
            logits[:] = np.nan
        return [logits[:, None]]


def test_accept_rejects_unavailable_evaluation(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference errors cannot fabricate a passing acceptance record."""
    import tools.ml_models.export.accept as accept

    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    monkeypatch.setattr(accept, "open_session", lambda *_args: Ok(_Session(invalid=True)))
    result = accept_artifact(tmp_path / "m.onnx", _manifest(), dataset)
    assert isinstance(result, Err)
    assert "finite" in result.error
    assert not acceptance_path(tmp_path / "m.onnx").exists()


def test_accept_measured_quality_failure(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete but wrong model produces a measured rejection, not an error."""
    import tools.ml_models.export.accept as accept

    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    monkeypatch.setattr(accept, "open_session", lambda *_args: Ok(_Session(inverted=True)))
    result = accept_artifact(tmp_path / "m.onnx", _manifest(), dataset, max_latency_ms=1e9)
    assert isinstance(result, Ok)
    assert result.value["accepted"] is False
    assert result.value["quality_ok"] is False


def test_accept_hash_check_before_sdk(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A wrong artifact hash never reaches SDK loading or scoring."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    artifact = tmp_path / "m.onnx"
    artifact.write_bytes(b"not an ONNX graph")
    result = accept_artifact(artifact, _manifest(), dataset)
    assert isinstance(result, Err)
    assert "hash" in result.error.lower() or "sha256" in result.error.lower()


class _EmptyMaskSession(_Session):
    def run(self, output_names: list[str] | None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.n += 1
        return [np.full_like(feeds["image"][:, :1], -4.0)]


def test_segmentor_acceptance_names_legacy_all_image_adapter(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One missed plume and one clean empty mask score 0.5 only under the legacy gate."""
    import tools.ml_models.export.accept as accept

    template = load_manifest(build_synthetic_dataset(tmp_path / "template", n=3) / "dataset.json")
    root = tmp_path / "mixed"
    writer = ShardWriter(
        root / "segmentor" / "test" / "2x3",
        2,
        2,
        3,
        channels=3,
        with_masks=True,
    )
    for index, label in enumerate((1.0, 0.0)):
        mask = np.zeros((1, 2, 3), dtype=np.uint8)
        mask[0, 0, 0] = int(label)
        writer.append(
            np.repeat(mask.astype(np.float32), 3, axis=0),
            np.array([16.0, 16.0], dtype=np.float32),
            label,
            mask,
            RowRecord(
                tile_id=str(index),
                group_id=str(index),
                frame_id=None,
                grid_rc=None,
                bin_id="",
                element="id",
            ),
        )
    writer.close()
    write_manifest(
        root / "dataset.json",
        replace(
            template,
            dataset_hash=compute_dataset_hash(root),
            shards=(
                ShardCount(task="segmentor", split="test", height=2, width=3, n=2, n_positive=1),
            ),
        ),
    )
    monkeypatch.setattr(accept, "open_session", lambda *_args: Ok(_EmptyMaskSession()))
    result = accept_artifact(
        tmp_path / "m.onnx",
        _manifest(kind="segmentor", arch="dilatenet", output_shape=(None, 1, None, None)),
        root,
        min_iou=0.4,
        max_latency_ms=1e9,
    )
    assert isinstance(result, Ok), result
    assert result.value["accepted"] is True
    assert result.value["metric"] == "foreground_iou_mean_all_annotated_images"
    assert result.value["quality_policy"] == "legacy_all_annotated_image_iou"
    evaluation = result.value["evaluation"]
    assert isinstance(evaluation, dict)
    metrics = evaluation["metrics"]
    assert isinstance(metrics, (tuple, list))
    values = {metric["name"]: metric["value"] for metric in metrics}
    assert values["foreground_iou_mean_all_annotated_images"] == 0.5
    assert values["foreground_iou_mean_positive_images"] == 0.0


def test_latency_rejection_is_measured_separately(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Perfect quality does not bypass the existing worst batch-one CPU limit."""
    import tools.ml_models.export.accept as accept

    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    monkeypatch.setattr(accept, "open_session", lambda *_args: Ok(_Session()))
    ticks = count()
    monkeypatch.setattr(
        accept,
        "time",
        SimpleNamespace(perf_counter=lambda: next(ticks) * 0.1),
    )
    result = accept_artifact(tmp_path / "m.onnx", _manifest(), dataset, max_latency_ms=50.0)
    assert isinstance(result, Ok)
    assert result.value["quality_ok"] is True
    assert result.value["latency_ok"] is False
    assert result.value["accepted"] is False
    assert result.value["worst_batch_one_latency_ms"] == pytest.approx(100.0)
