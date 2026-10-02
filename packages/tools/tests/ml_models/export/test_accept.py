"""Tests for acceptance gating that run without the onnxruntime SDK."""

from collections.abc import Callable
from pathlib import Path

import pytest
from tools.ml_models.export.accept import accept_artifact
from tools.ml_models.export.manifest import ModelManifest


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
            accept_artifact(tmp_path / "m.onnx", _manifest(), [tmp_path], **kwargs)


def test_accept_rejects_empty_datasets(tmp_path: Path) -> None:
    """At least one finished dataset is required."""
    with pytest.raises(ValueError):
        accept_artifact(tmp_path / "m.onnx", _manifest(), [])


def test_accept_rejects_reference_mismatch(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A dataset GSD reference that disagrees with the model is rejected."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=6)
    with pytest.raises(ValueError, match="disagree"):
        accept_artifact(
            tmp_path / "m.onnx",
            _manifest(gsd_reference_m=16.0),
            [dataset],
        )


def test_accept_reaches_session_open(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A compatible dataset passes preprocessing gates and fails at the SDK."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=6)
    artifact = tmp_path / "m.onnx"
    artifact.write_bytes(b"stub")
    manifest = _manifest()
    with pytest.raises(ValueError) as excinfo:
        accept_artifact(artifact, manifest, [dataset])
    assert "sha256" in str(excinfo.value) or "onnxruntime" in str(excinfo.value)
