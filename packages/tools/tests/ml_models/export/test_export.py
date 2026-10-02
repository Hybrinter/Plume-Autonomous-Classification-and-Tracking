"""Tests for two-input ONNX export gates that run without the onnx SDK.

Full export parity (real onnx + onnxruntime) lives in test_parity.py and
skips when the SDKs are absent.
"""

from pathlib import Path

import pytest
import torch
from flight.libs.types import Err
from tools.ml_models.arch.film import CONDITIONING_ID
from tools.ml_models.arch.registry import build
from tools.ml_models.export.export import ExportConfig, export

_FULL_COVERAGE = ((10.0, 10.0), (45.0, 45.0))
_PARTIAL_COVERAGE = ((16.0, 16.0), (17.0, 17.0))


def _checkpoint(
    path: Path,
    *,
    conditioning: str = CONDITIONING_ID,
    gsd_min_m: tuple[float, float] = _FULL_COVERAGE[0],
    gsd_max_m: tuple[float, float] = _FULL_COVERAGE[1],
    kind: str = "classifier",
    arch: str = "pactnet",
    norm: str = "unit",
    band_names: list[str] | None = None,
    gsd_reference_m: float = 15.87,
) -> Path:
    """Write a minimal training-format checkpoint for export tests."""
    model = build(kind, arch, 3)
    torch.save(
        {
            "kind": kind,
            "arch": arch,
            "state_dict": model.state_dict(),
            "epoch": 0,
            "conditioning": conditioning,
            "config": {},
            "provenance": {
                "in_channels": 3,
                "band_names": band_names or ["BLUE", "GREEN", "RED"],
                "norm": norm,
                "gsd_reference_m": gsd_reference_m,
                "gsd_min_m": list(gsd_min_m),
                "gsd_max_m": list(gsd_max_m),
                "train_samples": 8,
            },
            "dataset_weights": [1.0],
            "dataset_hash": "b" * 64,
            "input_height_px": 193,
            "input_width_px": 258,
        },
        path,
    )
    return path


def test_export_refuses_existing_artifact(tmp_path: Path) -> None:
    """An existing artifact path is rejected before any work."""
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"existing")
    result = export(ExportConfig(str(tmp_path / "missing.pt"), str(artifact)))
    assert isinstance(result, Err)
    assert artifact.read_bytes() == b"existing"


def test_export_refuses_existing_sidecar(tmp_path: Path) -> None:
    """An existing sidecar path is rejected before any work."""
    artifact = tmp_path / "model.onnx"
    (tmp_path / "model.json").write_text("{}")
    result = export(ExportConfig(str(tmp_path / "missing.pt"), str(artifact)))
    assert isinstance(result, Err)
    assert not artifact.exists()


def test_export_missing_checkpoint_is_err(tmp_path: Path) -> None:
    """A missing checkpoint returns Err rather than raising."""
    result = export(ExportConfig(str(tmp_path / "missing.pt"), str(tmp_path / "out.onnx")))
    assert isinstance(result, Err)


def test_export_rejects_unconditioned_checkpoint(tmp_path: Path) -> None:
    """Legacy 'ignored' conditioning is not exportable through this path."""
    ckpt = _checkpoint(tmp_path / "ckpt.pt", conditioning="ignored")
    result = export(ExportConfig(str(ckpt), str(tmp_path / "out.onnx")))
    assert isinstance(result, Err)
    assert "conditioning" in result.error


def test_export_rejects_partial_coverage(tmp_path: Path) -> None:
    """Training coverage short of the flight range fails without the override."""
    ckpt = _checkpoint(
        tmp_path / "ckpt.pt",
        gsd_min_m=_PARTIAL_COVERAGE[0],
        gsd_max_m=_PARTIAL_COVERAGE[1],
    )
    result = export(ExportConfig(str(ckpt), str(tmp_path / "out.onnx")))
    assert isinstance(result, Err)
    assert "coverage" in result.error


def test_export_partial_override_proceeds_to_onnx(tmp_path: Path) -> None:
    """allow_partial_gsd lets export proceed past the coverage gate."""
    pytest.importorskip("torch")
    ckpt = _checkpoint(
        tmp_path / "ckpt.pt",
        gsd_min_m=_PARTIAL_COVERAGE[0],
        gsd_max_m=_PARTIAL_COVERAGE[1],
    )
    result = export(ExportConfig(str(ckpt), str(tmp_path / "out.onnx"), allow_partial_gsd=True))
    # Without onnx installed this still fails at graph emission; with onnx it
    # succeeds and records partial_gsd. Either way the coverage gate passed.
    if isinstance(result, Err):
        assert "coverage" not in result.error


@pytest.mark.parametrize(
    "overrides",
    [
        {"norm": "dn"},
        {"band_names": ["RED", "GREEN", "BLUE"]},
        {"band_names": ["BLUE", "GREEN"]},
        {"gsd_reference_m": 12.0},
    ],
    ids=["norm-dn", "band-order", "band-count", "reference"],
)
def test_export_rejects_nonflight_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]
) -> None:
    """Raw-DN, wrong-band, or off-reference provenance fails before tracing."""
    ckpt = _checkpoint(tmp_path / "ckpt.pt", **overrides)  # type: ignore[arg-type]
    traced = False

    def _no_trace(*_args: object, **_kwargs: object) -> None:
        nonlocal traced
        traced = True

    monkeypatch.setattr(torch.onnx, "export", _no_trace)
    result = export(ExportConfig(str(ckpt), str(tmp_path / "out.onnx")))
    assert isinstance(result, Err)
    assert not traced


def test_export_never_overwrites_concurrent_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A destination created between precheck and publish is preserved."""
    import tools.ml_models.export.export as export_module

    ckpt = _checkpoint(tmp_path / "ckpt.pt")
    artifact = tmp_path / "model.onnx"

    def _fake_trace(_model: object, _args: object, path: object, **_kwargs: object) -> None:
        Path(str(path)).write_bytes(b"graph-bytes")
        # A foreign writer creates the destination during tracing.
        artifact.write_bytes(b"concurrent")

    def _fake_verify(
        _path: Path, _kind: str
    ) -> tuple[tuple[int | None, ...], tuple[int | None, ...], tuple[int | None, ...]]:
        return (None, 3, None, None), (None, 2), (None, 1)

    monkeypatch.setattr(torch.onnx, "export", _fake_trace)
    monkeypatch.setattr(export_module, "_refine_shape_metadata", lambda _path: None)
    monkeypatch.setattr(export_module, "_verify_graph", _fake_verify)
    result = export(ExportConfig(str(ckpt), str(artifact)))
    assert isinstance(result, Err)
    assert artifact.read_bytes() == b"concurrent"
    assert not (tmp_path / "model.json").exists()


def test_export_rolls_back_own_artifact_on_sidecar_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A foreign sidecar appearing after the precheck is preserved, and our
    just-linked artifact is rolled back."""
    import tools.ml_models.export.export as export_module

    ckpt = _checkpoint(tmp_path / "ckpt.pt")
    artifact = tmp_path / "model.onnx"
    sidecar = tmp_path / "model.json"

    def _fake_trace(_model: object, _args: object, path: object, **_kwargs: object) -> None:
        Path(str(path)).write_bytes(b"graph-bytes")

    def _fake_verify(
        _path: Path, _kind: str
    ) -> tuple[tuple[int | None, ...], tuple[int | None, ...], tuple[int | None, ...]]:
        # A foreign writer creates the sidecar destination during validation.
        sidecar.write_text('{"foreign": true}\n', encoding="utf-8")
        return (None, 3, None, None), (None, 2), (None, 1)

    monkeypatch.setattr(torch.onnx, "export", _fake_trace)
    monkeypatch.setattr(export_module, "_refine_shape_metadata", lambda _path: None)
    monkeypatch.setattr(export_module, "_verify_graph", _fake_verify)
    result = export(ExportConfig(str(ckpt), str(artifact)))
    assert isinstance(result, Err)
    assert sidecar.read_text() == '{"foreign": true}\n'
    assert not artifact.exists()
    assert not list(tmp_path.glob("*.partial"))
