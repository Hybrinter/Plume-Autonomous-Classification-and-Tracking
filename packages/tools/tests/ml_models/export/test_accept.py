"""Tests for the fail-closed acceptance boundary that run without onnxruntime."""

from collections.abc import Callable
from pathlib import Path

import pytest
from flight.libs.types import Err
from tools.ml_models.export.accept import accept_artifact
from tools.ml_models.export.manifest import ModelManifest, acceptance_path


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


def test_accept_scoring_unavailable_after_validation(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid inputs still fail closed: no session opens, no report is written."""
    import tools.ml_models.export.accept as accept

    dataset = build_synthetic_dataset(tmp_path / "ds", n=6)
    artifact = tmp_path / "m.onnx"
    artifact.write_bytes(b"stub")
    opened = False

    def _spy(*args: object, **kwargs: object) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("session opened while scoring is unavailable")

    monkeypatch.setattr(accept, "open_session", _spy)
    result = accept_artifact(artifact, _manifest(), dataset, min_accuracy=0.0, max_latency_ms=1e9)
    assert isinstance(result, Err)
    assert "unavailable" in result.error
    assert not opened
    assert not acceptance_path(artifact).exists()
