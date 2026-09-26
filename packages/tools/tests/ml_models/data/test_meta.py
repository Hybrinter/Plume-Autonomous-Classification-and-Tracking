"""Tests for dataset meta, provenance, and the pack hash."""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.data.meta import (
    DatasetMeta,
    Provenance,
    compute_dataset_hash,
    load_dataset_meta,
    load_provenance,
    provenance_from_meta,
    write_dataset_meta,
    write_provenance,
)
from tools.ml_models.data.norm import BandStats, apply_band_z, fit_band_stats
from tools.ml_models.data.split import SplitIndex, write_splits

_HASH_FILES = ("images.npy", "masks.npy", "labels.npy", "splits.json", "provenance.json")


def _sample_meta() -> DatasetMeta:
    """Return a sidecar that sets every field, including provenance."""
    return DatasetMeta(
        dataset_hash="abc123",
        source_doi="10.5281/zenodo.4250706",
        n=4,
        height=8,
        width=6,
        in_channels=2,
        band_names=("B04", "B08"),
        norm="band_z",
        bit_depth=12,
        ingest_path="sentinel2_4250706_prism_proxy",
        radiometry="s2_l2a_reflectance",
        gsd_m=10.5,
        extent_m=120.0,
        weight_table_id="s2-l2a-v1",
        band_mean=(10.0, 20.0),
        band_std=(1.5, 2.5),
    )


def _write_hash_inputs(root: Path, meta: DatasetMeta) -> None:
    """Write the five hashed pack files for ``meta``."""
    root.mkdir(parents=True, exist_ok=True)
    images = np.zeros((meta.n, meta.in_channels, meta.height, meta.width), dtype=np.float32)
    masks = np.zeros((meta.n, 1, meta.height, meta.width), dtype=np.float32)
    labels = np.zeros((meta.n, 1), dtype=np.float32)
    np.save(root / "images.npy", images)
    np.save(root / "masks.npy", masks)
    np.save(root / "labels.npy", labels)
    write_splits(
        root / "splits.json",
        SplitIndex(train=tuple(range(meta.n - 2)), val=(meta.n - 2,), test=(meta.n - 1,)),
    )
    write_provenance(root / "provenance.json", provenance_from_meta(meta))


def _rewrite_json(path: Path, updates: dict[str, object]) -> None:
    """Replace keys in a sidecar file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manual_hash(root: Path) -> str:
    """Hash ``name:file_sha256`` lines the same way the pack codec does."""
    outer = hashlib.sha256()
    for name in _HASH_FILES:
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        outer.update(f"{name}:{digest}\n".encode())
    return outer.hexdigest()


def test_dataset_meta_round_trip_keeps_every_field(tmp_path: Path) -> None:
    """write_dataset_meta / load_dataset_meta keep every provenance field."""
    meta = _sample_meta()
    path = tmp_path / "dataset.json"
    write_dataset_meta(path, meta)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "dataset_hash": "abc123",
        "source_doi": "10.5281/zenodo.4250706",
        "n": 4,
        "height": 8,
        "width": 6,
        "in_channels": 2,
        "band_names": ["B04", "B08"],
        "norm": "band_z",
        "bit_depth": 12,
        "ingest_path": "sentinel2_4250706_prism_proxy",
        "radiometry": "s2_l2a_reflectance",
        "gsd_m": 10.5,
        "extent_m": 120.0,
        "weight_table_id": "s2-l2a-v1",
        "band_mean": [10.0, 20.0],
        "band_std": [1.5, 2.5],
    }
    assert load_dataset_meta(path) == meta


def test_provenance_round_trip(tmp_path: Path) -> None:
    """write_provenance / load_provenance keep every provenance field."""
    provenance = provenance_from_meta(_sample_meta())
    path = tmp_path / "provenance.json"
    write_provenance(path, provenance)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["band_mean"] == [10.0, 20.0]
    assert payload["band_std"] == [1.5, 2.5]
    assert load_provenance(path) == provenance


def test_band_z_provenance_requires_moments() -> None:
    """band_z without one moment per band raises ValueError."""
    with pytest.raises(ValueError, match="band_mean"):
        Provenance(
            ingest_path="flight_camera",
            radiometry="normalize_dn",
            gsd_m=10.0,
            extent_m=20.0,
            weight_table_id="table-a",
            band_names=("b0", "b1"),
            norm="band_z",
            bit_depth=12,
        )


@pytest.mark.parametrize(
    ("band_mean", "band_std"),
    [
        ((math.nan,), (1.0,)),
        ((math.inf,), (1.0,)),
        ((-math.inf,), (1.0,)),
        ((0.0,), (math.nan,)),
        ((0.0,), (math.inf,)),
        ((0.0,), (-math.inf,)),
        ((0.0,), (0.0,)),
    ],
)
def test_band_z_provenance_rejects_non_finite_moments(
    band_mean: tuple[float, ...],
    band_std: tuple[float, ...],
) -> None:
    """band_z rejects NaN and infinite means and standard deviations."""
    with pytest.raises(ValueError, match="finite"):
        Provenance(
            ingest_path="flight_camera",
            radiometry="normalize_dn",
            gsd_m=10.0,
            extent_m=20.0,
            weight_table_id="table-a",
            band_names=("b0",),
            norm="band_z",
            bit_depth=12,
            band_mean=band_mean,
            band_std=band_std,
        )


def test_other_norms_reject_stored_moments() -> None:
    """normalize_dn and unit keep empty moment lists."""
    with pytest.raises(ValueError, match="empty"):
        Provenance(
            ingest_path="flight_camera",
            radiometry="normalize_dn",
            gsd_m=10.0,
            extent_m=20.0,
            weight_table_id="table-a",
            band_names=("b0",),
            norm="unit",
            bit_depth=12,
            band_mean=(0.0,),
            band_std=(1.0,),
        )


def test_loaded_band_moments_reproduce_band_z(tmp_path: Path) -> None:
    """Loaded mean and std rebuild the z-score for a new sample."""
    train = np.array([[[0.0, 2.0], [4.0, 6.0]], [[1.0, 1.0], [1.0, 1.0]]], dtype=np.float32)
    stats = fit_band_stats(train)
    provenance = Provenance(
        ingest_path="flight_camera",
        radiometry="normalize_dn",
        gsd_m=10.0,
        extent_m=20.0,
        weight_table_id="table-a",
        band_names=("b0", "b1"),
        norm="band_z",
        bit_depth=12,
        band_mean=stats.mean,
        band_std=stats.std,
    )
    path = tmp_path / "provenance.json"
    write_provenance(path, provenance)
    loaded = load_provenance(path)
    sample = np.array([[[3.0, 5.0], [7.0, 9.0]], [[2.0, 2.0], [2.0, 2.0]]], dtype=np.float32)
    restored = BandStats(mean=loaded.band_mean, std=loaded.band_std)
    np.testing.assert_array_equal(apply_band_z(sample, restored), apply_band_z(sample, stats))


def test_dataset_hash_covers_band_moments(tmp_path: Path) -> None:
    """A change to fitted moments changes the pack hash."""
    root = tmp_path / "pack"
    meta = _sample_meta()
    _write_hash_inputs(root, meta)
    first = compute_dataset_hash(root)
    shifted = replace(meta, band_mean=(0.0, 1.0))
    write_provenance(root / "provenance.json", provenance_from_meta(shifted))
    assert compute_dataset_hash(root) != first


def test_loaders_reject_bool_bit_depth(tmp_path: Path) -> None:
    """JSON true for bit_depth is rejected rather than coerced to 1."""
    meta = _sample_meta()
    dataset_path = tmp_path / "dataset.json"
    provenance_path = tmp_path / "provenance.json"
    write_dataset_meta(dataset_path, meta)
    write_provenance(provenance_path, provenance_from_meta(meta))
    _rewrite_json(dataset_path, {"bit_depth": True})
    _rewrite_json(provenance_path, {"bit_depth": True})
    with pytest.raises(ValueError, match="bit_depth"):
        load_dataset_meta(dataset_path)
    with pytest.raises(ValueError, match="bit_depth"):
        load_provenance(provenance_path)


def test_loaders_reject_numeric_strings(tmp_path: Path) -> None:
    """A numeric string does not become an int or a float."""
    meta = _sample_meta()
    dataset_path = tmp_path / "dataset.json"
    provenance_path = tmp_path / "provenance.json"
    write_dataset_meta(dataset_path, meta)
    write_provenance(provenance_path, provenance_from_meta(meta))
    _rewrite_json(dataset_path, {"bit_depth": "12"})
    _rewrite_json(provenance_path, {"gsd_m": "10.5"})
    with pytest.raises(ValueError, match="bit_depth"):
        load_dataset_meta(dataset_path)
    with pytest.raises(ValueError, match="gsd_m"):
        load_provenance(provenance_path)


def test_loaders_keep_json_numbers_and_arrays(tmp_path: Path) -> None:
    """A normal sidecar loads, including a JSON integer for a float field."""
    meta = _sample_meta()
    dataset_path = tmp_path / "dataset.json"
    provenance_path = tmp_path / "provenance.json"
    write_dataset_meta(dataset_path, meta)
    write_provenance(provenance_path, provenance_from_meta(meta))
    _rewrite_json(dataset_path, {"gsd_m": 10, "band_mean": [10, 20]})
    _rewrite_json(provenance_path, {"extent_m": 120, "band_std": [1, 2]})
    loaded_meta = load_dataset_meta(dataset_path)
    loaded_provenance = load_provenance(provenance_path)
    assert loaded_meta.gsd_m == 10.0
    assert loaded_meta.band_mean == (10.0, 20.0)
    assert loaded_meta.band_names == ("B04", "B08")
    assert loaded_meta.bit_depth == 12
    assert loaded_provenance.extent_m == 120.0
    assert loaded_provenance.band_std == (1.0, 2.0)
    assert loaded_provenance.band_names == ("B04", "B08")


def test_load_dataset_meta_rejects_unknown_key(tmp_path: Path) -> None:
    """An unknown dataset.json key raises ValueError."""
    path = tmp_path / "dataset.json"
    write_dataset_meta(path, _sample_meta())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["note"] = "extra"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="extra"):
        load_dataset_meta(path)


def test_dataset_hash_matches_name_digest_lines(tmp_path: Path) -> None:
    """The pack hash is the SHA-256 of name:file_sha256 lines."""
    root = tmp_path / "pack"
    _write_hash_inputs(root, _sample_meta())
    assert compute_dataset_hash(root) == _manual_hash(root)


def test_dataset_hash_ignores_dataset_json(tmp_path: Path) -> None:
    """dataset.json is not an input to the pack hash."""
    root = tmp_path / "pack"
    meta = _sample_meta()
    _write_hash_inputs(root, meta)
    digest = compute_dataset_hash(root)
    stored = replace(meta, dataset_hash=digest)
    write_dataset_meta(root / "dataset.json", stored)
    text = (root / "dataset.json").read_text(encoding="utf-8")
    (root / "dataset.json").write_text(text.replace("4250706", "0000000"), encoding="utf-8")
    assert compute_dataset_hash(root) == digest


def test_load_dataset_meta_rejects_hash_mismatch(tmp_path: Path) -> None:
    """verify=True raises ValueError when the pack bytes change."""
    root = tmp_path / "pack"
    meta = _sample_meta()
    _write_hash_inputs(root, meta)
    stored = replace(meta, dataset_hash=compute_dataset_hash(root))
    write_dataset_meta(root / "dataset.json", stored)
    assert load_dataset_meta(root / "dataset.json", pack_dir=root) == stored
    labels = np.load(root / "labels.npy")
    labels = labels.copy()
    labels[0, 0] = 1.0
    np.save(root / "labels.npy", labels)
    with pytest.raises(ValueError, match="hash"):
        load_dataset_meta(root / "dataset.json", pack_dir=root)
    assert load_dataset_meta(root / "dataset.json", pack_dir=root, verify=False) == stored


def test_compute_dataset_hash_requires_provenance(tmp_path: Path) -> None:
    """A missing provenance.json raises FileNotFoundError."""
    root = tmp_path / "pack"
    _write_hash_inputs(root, _sample_meta())
    (root / "provenance.json").unlink()
    with pytest.raises(FileNotFoundError, match="provenance.json"):
        compute_dataset_hash(root)
