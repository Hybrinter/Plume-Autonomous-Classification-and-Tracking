"""Tests for promotion gating and the classifier/segmentor pair manifest."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tools.ml_models.export.pair as pair_module
from flight.libs.types import Err, Ok, Result
from tools.ml_models.export.manifest import ModelManifest, write_manifest
from tools.ml_models.export.pair import (
    flight_promotable,
    training_promotable,
    write_pair_manifest,
)
from tools.ml_models.export.session import Node, Session

_FULL_MIN = (15.0, 15.0)
_FULL_MAX = (40.0, 40.0)
_PARTIAL_MIN = (16.0, 16.0)
_PARTIAL_MAX = (17.0, 17.0)


def _summary(**overrides: object) -> dict[str, object]:
    """A promotable run summary; overrides patch top-level keys."""
    payload: dict[str, object] = {
        "kind": "classifier",
        "arch": "pactnet",
        "conditioning": "film-log-gsd-v1",
        "dataset_hash": "c" * 64,
        "provenance": {
            "band_names": ["BLUE", "GREEN", "RED"],
            "norm": "unit",
            "gsd_reference_m": 15.87,
            "gsd_min_m": list(_FULL_MIN),
            "gsd_max_m": list(_FULL_MAX),
            "spatial_shapes": [[120, 120]],
            "train_samples": 8,
        },
    }
    provenance_raw = payload["provenance"]
    assert isinstance(provenance_raw, dict)
    provenance = dict(provenance_raw)
    for key, value in overrides.items():
        if key in provenance:
            provenance[key] = value
        else:
            payload[key] = value
    payload["provenance"] = provenance
    return payload


def _run(tmp_path: Path, **overrides: object) -> Path:
    """Create a run directory with summary.json and checkpoints/last.pt."""
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "last.pt").write_bytes(b"stub")
    (run / "summary.json").write_text(json.dumps(_summary(**overrides)))
    return run


def test_training_promotable_native_only_full_coverage(tmp_path: Path) -> None:
    """A native-extent run with full GSD coverage is promotable."""
    assert training_promotable(_run(tmp_path))


def test_training_promotable_missing_artifacts(tmp_path: Path) -> None:
    """Missing summary or checkpoint fails the gate."""
    run = _run(tmp_path)
    (run / "summary.json").unlink()
    assert not training_promotable(run)
    run2 = _run(tmp_path / "other")
    (run2 / "checkpoints" / "last.pt").unlink()
    assert not training_promotable(run2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"arch": "dilatenet"},
        {"kind": "detector"},
        {"conditioning": "ignored"},
        {"band_names": ["RED", "GREEN", "BLUE"]},
        {"norm": "dn"},
        {"gsd_reference_m": 12.0},
        {"gsd_min_m": [16.0, 16.0], "gsd_max_m": [17.0, 17.0]},
    ],
    ids=[
        "arch-family",
        "kind",
        "conditioning",
        "bands",
        "norm",
        "reference",
        "coverage",
    ],
)
def test_training_promotable_rejects_contract_violations(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    """Family, preprocessing, and coverage mismatches all fail."""
    assert not training_promotable(_run(tmp_path, **overrides))


def test_training_promotable_partial_coverage_override(tmp_path: Path) -> None:
    """allow_partial_gsd admits coverage-short runs explicitly."""
    run = _run(tmp_path, gsd_min_m=[16.0, 16.0], gsd_max_m=[17.0, 17.0])
    assert not training_promotable(run)
    assert training_promotable(run, allow_partial_gsd=True)


def _sidecar(tmp_path: Path, name: str, kind: str, arch: str) -> tuple[Path, ModelManifest]:
    """Write a dummy artifact plus a valid sidecar; return (sidecar, manifest)."""
    artifact = tmp_path / f"{name}.onnx"
    artifact.write_bytes(f"{name}-bytes".encode())
    manifest = ModelManifest(
        version="a" * 16,
        kind=kind,  # type: ignore[arg-type]
        arch=arch,
        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        dataset_hash="b" * 64,
        band_names=("BLUE", "GREEN", "RED"),
        input_shape=(None, 3, 193, 258),
        gsd_input_shape=(None, 2),
        output_shape=(None, 1) if kind == "classifier" else (None, 1, None, None),
        gsd_reference_m=15.87,
        gsd_min_m=_FULL_MIN,
        gsd_max_m=_FULL_MAX,
        conditioning="film-log-gsd-v1",
        gsd_encoding="ln_metres_over_reference_lateral_along",
    )
    sidecar = tmp_path / f"{name}.json"
    write_manifest(sidecar, manifest)
    return sidecar, manifest


def _acceptance(artifact: Path, sha256: str, accepted: bool = True) -> None:
    """Write a sibling acceptance report for an artifact."""
    artifact.with_suffix(".acceptance.json").write_text(
        json.dumps({"accepted": accepted, "sha256": sha256})
    )


def test_pair_rejects_wrong_kinds(tmp_path: Path) -> None:
    """Swapped or repeated kinds are rejected before session checks."""
    cls, _ = _sidecar(tmp_path, "cls", "segmentor", "dilatenet")
    seg, _ = _sidecar(tmp_path, "seg", "segmentor", "dilatenet")
    result = write_pair_manifest(cls, seg, tmp_path / "pair.json")
    assert isinstance(result, Err)


def test_pair_rejects_missing_acceptance(tmp_path: Path) -> None:
    """Artifacts without an acceptance report never pair."""
    cls, _ = _sidecar(tmp_path, "cls", "classifier", "pactnet")
    seg, _ = _sidecar(tmp_path, "seg", "segmentor", "dilatenet")
    result = write_pair_manifest(cls, seg, tmp_path / "pair.json")
    assert isinstance(result, Err)
    assert "acceptance" in result.error


def test_pair_rejects_failed_or_mismatched_acceptance(tmp_path: Path) -> None:
    """Failed or hash-mismatched reports are not evidence."""
    cls, cls_manifest = _sidecar(tmp_path, "cls", "classifier", "pactnet")
    seg, seg_manifest = _sidecar(tmp_path, "seg", "segmentor", "dilatenet")
    _acceptance(tmp_path / "cls.onnx", cls_manifest.sha256, accepted=False)
    _acceptance(tmp_path / "seg.onnx", seg_manifest.sha256)
    result = write_pair_manifest(cls, seg, tmp_path / "pair.json")
    assert isinstance(result, Err)

    _acceptance(tmp_path / "cls.onnx", cls_manifest.sha256)
    _acceptance(tmp_path / "seg.onnx", "0" * 64)
    result = write_pair_manifest(cls, seg, tmp_path / "pair.json")
    assert isinstance(result, Err)


def test_pair_rejects_missing_sidecar(tmp_path: Path) -> None:
    """Missing sidecar files surface as Err."""
    result = write_pair_manifest(
        tmp_path / "none1.json", tmp_path / "none2.json", tmp_path / "pair.json"
    )
    assert isinstance(result, Err)


def _run_sidecar(run: Path, **overrides: object) -> Path:
    """Write a dummy artifact and matching sidecar inside a run directory."""
    artifact = run / "model.onnx"
    artifact.write_bytes(b"graph-bytes")
    fields: dict[str, object] = {
        "version": "a" * 16,
        "kind": "classifier",
        "arch": "pactnet",
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "dataset_hash": "c" * 64,
        "band_names": ("BLUE", "GREEN", "RED"),
        "input_shape": (None, 3, 193, 258),
        "gsd_input_shape": (None, 2),
        "output_shape": (None, 1),
        "gsd_reference_m": 15.87,
        "gsd_min_m": _FULL_MIN,
        "gsd_max_m": _FULL_MAX,
        "conditioning": "film-log-gsd-v1",
        "gsd_encoding": "ln_metres_over_reference_lateral_along",
    }
    fields.update(overrides)
    manifest = ModelManifest(**fields)  # type: ignore[arg-type]
    write_manifest(run / "model.json", manifest)
    return artifact


def test_flight_promotable_requires_a_real_graph(tmp_path: Path) -> None:
    """Metadata alone is never promotable; a validated graph is required."""
    run = _run(tmp_path)
    assert training_promotable(run)
    assert not flight_promotable(run)


def test_flight_promotable_strict_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A matching artifact with a valid session passes the strict gate."""
    run = _run(tmp_path)
    _run_sidecar(run)
    monkeypatch.setattr(pair_module, "open_session", lambda *_a, **_k: Ok(object()))
    monkeypatch.setattr(pair_module, "_flight_loader_ok", lambda *_a, **_k: True)
    assert flight_promotable(run)


def test_flight_promotable_rejects_dynamic_research_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dynamic spatial research artifacts are not flight eligible."""
    run = _run(tmp_path)
    _run_sidecar(run, input_shape=(None, 3, None, None))
    monkeypatch.setattr(pair_module, "open_session", lambda *_a, **_k: Ok(object()))
    monkeypatch.setattr(pair_module, "_flight_loader_ok", lambda *_a, **_k: False)
    assert not flight_promotable(run)


def test_flight_promotable_rejects_sdk_absent(tmp_path: Path) -> None:
    """Without a loadable session the strict gate fails even for valid files."""
    run = _run(tmp_path)
    _run_sidecar(run)
    # Real open_session: onnxruntime is absent in this environment.
    assert not flight_promotable(run)


def test_flight_promotable_rejects_metadata_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sidecar disagreeing with the run metadata fails despite a good session."""
    run = _run(tmp_path)
    _run_sidecar(run, dataset_hash="d" * 64)
    monkeypatch.setattr(pair_module, "open_session", lambda *_a, **_k: Ok(object()))
    assert not flight_promotable(run)


@pytest.mark.parametrize(
    ("inputs", "outputs"),
    [
        (
            [
                Node("image", ("batch", 3, "h", "w"), "tensor(float)"),
                Node("gsd", ("batch", 3), "tensor(float)"),
            ],
            [Node("logits", ("batch", 1), "tensor(float)")],
        ),
        (
            [
                Node("image", ("batch", 3, "h", "w"), "tensor(float)"),
                Node("scale", ("batch", 2), "tensor(float)"),
            ],
            [Node("logits", ("batch", 1), "tensor(float)")],
        ),
        (
            [
                Node("image", (1, 3, 193, 258), "tensor(float)"),
                Node("gsd", (1, 2), "tensor(float)"),
            ],
            [Node("logits", (1, 1), "tensor(float)")],
        ),
    ],
    ids=["gsd-width-3", "wrong-input-name", "fixed-batch"],
)
def test_flight_promotable_rejects_bad_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inputs: list[Node],
    outputs: list[Node],
) -> None:
    """The strict gate runs the real contract checks on the declared graph."""
    from tools.ml_models.export.session import open_session as real_open_session

    class _BadSession:
        def get_inputs(self) -> list[Node]:
            return inputs

        def get_outputs(self) -> list[Node]:
            return outputs

        def run(
            self, output_names: list[str] | None, feeds: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            return []

    def _open(
        artifact: str | Path, manifest: ModelManifest, **_kwargs: object
    ) -> Result[Session, str]:
        return real_open_session(artifact, manifest, loader=lambda _path: _BadSession())

    run = _run(tmp_path)
    _run_sidecar(run)
    monkeypatch.setattr(pair_module, "open_session", _open)
    assert not flight_promotable(run)


def test_flight_promotable_artifact_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit artifact path outside the run directory is honored."""
    run = _run(tmp_path)
    elsewhere = tmp_path / "deploy"
    elsewhere.mkdir()
    artifact = elsewhere / "model.onnx"
    artifact.write_bytes(b"graph-bytes")
    manifest = ModelManifest(
        version="a" * 16,
        kind="classifier",
        arch="pactnet",
        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        dataset_hash="c" * 64,
        band_names=("BLUE", "GREEN", "RED"),
        input_shape=(None, 3, 193, 258),
        gsd_input_shape=(None, 2),
        output_shape=(None, 1),
        gsd_reference_m=15.87,
        gsd_min_m=_FULL_MIN,
        gsd_max_m=_FULL_MAX,
        conditioning="film-log-gsd-v1",
        gsd_encoding="ln_metres_over_reference_lateral_along",
    )
    write_manifest(elsewhere / "model.json", manifest)
    monkeypatch.setattr(pair_module, "open_session", lambda *_a, **_k: Ok(object()))
    monkeypatch.setattr(pair_module, "_flight_loader_ok", lambda *_a, **_k: True)
    assert flight_promotable(run, artifact_path=artifact)


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": 42},
        {"band_names": 5},
        {"gsd_min_m": "bad"},
        {"gsd_min_m": [float("nan"), 15.0]},
        {"gsd_min_m": [50.0, 14.0], "gsd_max_m": [40.0, 40.0]},
    ],
    ids=["kind-int", "bands-int", "gsd-str", "gsd-nan", "gsd-unordered"],
)
def test_training_promotable_malformed_metadata(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    """Malformed metadata returns False — never raises, even with the override."""
    run = _run(tmp_path, **overrides)
    assert not training_promotable(run, allow_partial_gsd=True)
    assert not flight_promotable(run, allow_partial_gsd=True)
