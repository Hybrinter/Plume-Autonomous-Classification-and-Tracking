"""Tests for the schema-2 model sidecar."""

import json
from pathlib import Path

import pytest
from tools.ml_models.export.manifest import (
    ModelManifest,
    acceptance_path,
    load_manifest,
    sidecar_path,
    write_manifest,
)

_SHA = "a" * 64
_HASH = "b" * 64


def _manifest(**overrides: object) -> ModelManifest:
    """Build a valid dynamic-spatial classifier manifest with overrides."""
    fields: dict[str, object] = {
        "version": "a" * 16,
        "kind": "classifier",
        "arch": "pactnet",
        "sha256": _SHA,
        "dataset_hash": _HASH,
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


def test_valid_manifest_constructs() -> None:
    """A conforming manifest passes validation and serializes."""
    manifest = _manifest()
    assert manifest.schema == 2
    assert manifest.quantization == "fp32"
    assert manifest.tile_hw == (193, 258)
    assert manifest.partial_gsd is False


def test_segmentor_manifest_accepts_dynamic_output() -> None:
    """Segmentor manifests declare (None, 1, H, W)-style outputs."""
    manifest = _manifest(
        kind="segmentor",
        arch="dilatenet_w32_s8",
        output_shape=(None, 1, None, None),
    )
    assert manifest.kind == "segmentor"


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": 1},
        {"arch": "dilatenet"},
        {"conditioning": "none"},
        {"gsd_encoding": "raw"},
        {"band_names": ("RED", "GREEN", "BLUE")},
        {"gsd_reference_m": 0.0},
        {"gsd_reference_m": float("nan")},
        {"gsd_min_m": (0.0, 14.0)},
        {"gsd_min_m": (50.0, 14.0)},
        {"sha256": "abc"},
        {"sha256": "z" * 64},
        {"dataset_hash": "abc"},
        {"input_shape": (1, 3, 193, 258)},
        {"gsd_input_shape": (None, 3)},
        {"output_shape": (None, 2)},
        {"tile_hw": (120, 120)},
        {"grid": (4, 4)},
        {"frame_hw": (100, 100)},
        {"input_types": {"image": "float32"}},
        {"input_types": {"image": "float32", "gsd": "float64"}},
        {"output_type": "int8"},
        {"norm": "dn"},
        {"norm": "uint16"},
    ],
    ids=[
        "schema-1",
        "arch-kind-mismatch",
        "conditioning",
        "encoding",
        "band-order",
        "ref-zero",
        "ref-nan",
        "gsd-min-zero",
        "gsd-unordered",
        "sha-short",
        "sha-nonhex",
        "dataset-hash-short",
        "fixed-batch",
        "gsd-shape",
        "output-shape",
        "tile-hw",
        "grid",
        "frame-hw",
        "input-types-missing-gsd",
        "input-types-wrong-dtype",
        "output-type-int8",
        "norm-dn",
        "norm-uint16",
    ],
)
def test_manifest_rejects_contract_violations(overrides: dict[str, object]) -> None:
    """Every contract violation raises a validation error."""
    with pytest.raises(ValueError):
        _manifest(**overrides)


def test_manifest_rejects_extra_fields() -> None:
    """Unknown sidecar keys are forbidden."""
    fields = {
        "version": "a" * 16,
        "kind": "classifier",
        "arch": "pactnet",
        "sha256": _SHA,
        "dataset_hash": _HASH,
        "band_names": ["BLUE", "GREEN", "RED"],
        "input_shape": [None, 3, None, None],
        "gsd_input_shape": [None, 2],
        "output_shape": [None, 1],
        "gsd_reference_m": 15.87,
        "gsd_min_m": [14.0, 14.0],
        "gsd_max_m": [40.0, 40.0],
        "conditioning": "film-log-gsd-v1",
        "gsd_encoding": "ln_metres_over_reference_lateral_along",
        "unexpected": 1,
    }
    with pytest.raises(ValueError):
        load_manifest_from_dict(fields)


def load_manifest_from_dict(payload: dict[str, object]) -> ModelManifest:
    """Validate a raw payload the way ``load_manifest`` does."""
    from pydantic import TypeAdapter

    return TypeAdapter(ModelManifest).validate_python(payload)


def test_sidecar_roundtrip(tmp_path: Path) -> None:
    """write_manifest then load_manifest reproduces the dataclass."""
    manifest = _manifest(kind="segmentor", arch="dilatenet", output_shape=(None, 1, 193, 258))
    dest = tmp_path / "model.json"
    write_manifest(dest, manifest)
    loaded = load_manifest(dest)
    assert loaded == manifest
    raw = json.loads(dest.read_text())
    assert raw["schema"] == 2
    assert raw["input_shape"] == [None, 3, None, None]


def test_sidecar_paths(tmp_path: Path) -> None:
    """Sidecar and acceptance paths are siblings of the artifact."""
    artifact = tmp_path / "model.onnx"
    assert sidecar_path(artifact) == tmp_path / "model.json"
    assert acceptance_path(artifact) == tmp_path / "model.acceptance.json"


def test_load_manifest_rejects_malformed(tmp_path: Path) -> None:
    """Missing files and non-object JSON raise."""
    missing = tmp_path / "none.json"
    with pytest.raises(OSError):
        load_manifest(missing)
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    with pytest.raises(ValueError):
        load_manifest(bad)
