"""Tests for acceptance gating that run without the onnxruntime SDK."""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from flight.libs.types import Ok
from tools.ml_models.dataset.gsd import to_model_gsd
from tools.ml_models.dataset.manifest import load_manifest
from tools.ml_models.dataset.store import read_gsd, read_images
from tools.ml_models.export.accept import accept_artifact
from tools.ml_models.export.manifest import ModelManifest
from tools.ml_models.export.session import Node, Session


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
    """Out-of-range thresholds raise before any dataset or model loads."""
    for kwargs in (
        {"min_iou": -0.1},
        {"min_iou": 1.1},
        {"min_accuracy": float("nan")},
        {"max_latency_ms": 0.0},
    ):
        with pytest.raises(ValueError, match="threshold"):
            accept_artifact(tmp_path / "m.onnx", _manifest(), tmp_path, **kwargs)


def test_accept_rejects_reference_mismatch(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A dataset GSD reference that disagrees with the model is rejected."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=6)
    with pytest.raises(ValueError, match="disagree"):
        accept_artifact(
            tmp_path / "m.onnx",
            _manifest(gsd_reference_m=16.0),
            dataset,
        )


def test_accept_rejects_band_mismatch(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A dataset band order that disagrees with the model is rejected."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=6)
    manifest = _manifest()
    object.__setattr__(manifest, "band_names", ("A", "B", "C"))
    with pytest.raises(ValueError, match="disagree"):
        accept_artifact(
            tmp_path / "m.onnx",
            manifest,
            dataset,
        )


def test_accept_rejects_empty_dataset(tmp_path: Path) -> None:
    """An empty dataset path is rejected before any loading."""
    with pytest.raises(ValueError, match="exactly one finished dataset"):
        accept_artifact(tmp_path / "m.onnx", _manifest(), "")


def test_accept_rejects_fixed_input_shape_mismatch(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixed model H/W mismatch fails before the artifact session opens."""
    import tools.ml_models.export.accept as accept

    dataset = build_synthetic_dataset(tmp_path / "ds", n=6)
    artifact = tmp_path / "m.onnx"
    artifact.write_bytes(b"stub")

    def _boom(artifact: Path, manifest: ModelManifest) -> Ok[Session]:
        raise AssertionError("session opened before the shape check")

    monkeypatch.setattr(accept, "open_session", _boom)
    for shape in ((None, 3, 120, 258), (None, 3, 193, 120), (None, 3, 9, 9)):
        manifest = _manifest()
        object.__setattr__(manifest, "input_shape", shape)
        with pytest.raises(ValueError, match="input_shape"):
            accept_artifact(artifact, manifest, dataset)


def test_accept_feeds_exact_stored_pixels_and_encoded_gsd(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session receives stored float32 pixels and encoded GSD unchanged."""
    import tools.ml_models.export.accept as accept

    dataset = build_synthetic_dataset(tmp_path / "ds", n=6)
    artifact = tmp_path / "m.onnx"
    artifact.write_bytes(b"stub")
    feeds: list[dict[str, np.ndarray]] = []

    class _SpySession:
        def get_inputs(self) -> Sequence[Node]:
            return [
                Node("image", ("batch", 3, "height", "width"), "tensor(float)"),
                Node("gsd", ("batch", 2), "tensor(float)"),
            ]

        def get_outputs(self) -> Sequence[Node]:
            return [Node("logits", ("batch", 1), "tensor(float)")]

        def run(
            self, output_names: list[str] | None, feed: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            feeds.append({name: value.copy() for name, value in feed.items()})
            return [np.zeros((feed["image"].shape[0], 1), dtype=np.float32)]

    def fake_open(artifact: Path, manifest: ModelManifest) -> Ok[Session]:
        return Ok(_SpySession())

    monkeypatch.setattr(accept, "open_session", fake_open)
    manifest = _manifest()
    report = accept_artifact(artifact, manifest, dataset, min_accuracy=0.0, max_latency_ms=1e9)
    assert report["accepted"] is True
    assert feeds
    dataset_manifest = load_manifest(dataset / "dataset.json")
    expected_images = np.concatenate(
        [
            read_images(dataset / "classifier" / "test" / f"{shard.height}x{shard.width}")
            for shard in dataset_manifest.shards
            if shard.task == "classifier" and shard.split == "test"
        ]
    )
    expected_gsd = np.concatenate(
        [
            to_model_gsd(
                read_gsd(dataset / "classifier" / "test" / f"{shard.height}x{shard.width}"),
                dataset_manifest.gsd_reference_m,
            ).astype(np.float32)
            for shard in dataset_manifest.shards
            if shard.task == "classifier" and shard.split == "test"
        ]
    )
    np.testing.assert_array_equal(
        np.concatenate([feed["image"] for feed in feeds]), expected_images
    )
    np.testing.assert_array_equal(np.concatenate([feed["gsd"] for feed in feeds]), expected_gsd)


def test_accept_reaches_session_open(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A compatible dataset passes preprocessing gates and fails at the SDK."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=6)
    artifact = tmp_path / "m.onnx"
    artifact.write_bytes(b"stub")
    manifest = _manifest()
    with pytest.raises(ValueError) as excinfo:
        accept_artifact(artifact, manifest, dataset)
    assert "sha256" in str(excinfo.value) or "onnxruntime" in str(excinfo.value)
