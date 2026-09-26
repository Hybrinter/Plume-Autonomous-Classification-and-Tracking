"""Tests for processed-pack write, load, and concatenation."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.data.meta import (
    IngestPath,
    NormName,
    Provenance,
    compute_dataset_hash,
    dataset_meta_from_provenance,
    load_dataset_meta,
    provenance_from_meta,
    write_dataset_meta,
)
from tools.ml_models.data.pack import (
    ProcessedPack,
    assert_same_ingest,
    concat_packs,
    load_processed_pack,
    write_processed_pack,
)
from tools.ml_models.data.split import SplitIndex, SplitRecipe

_SPLIT3 = SplitIndex(train=(0,), val=(1,), test=(2,))


def _provenance(
    ingest: IngestPath = "flight_camera",
    *,
    bands: tuple[str, ...] = ("b0",),
    norm: NormName = "unit",
    gsd_m: float = 10.5,
) -> Provenance:
    """Return a provenance record for a one-band pack."""
    return Provenance(
        ingest_path=ingest,
        radiometry="normalize_dn",
        gsd_m=gsd_m,
        extent_m=20.0,
        weight_table_id="table-a",
        band_names=bands,
        norm=norm,
        bit_depth=12,
    )


def _arrays(
    n: int,
    *,
    channels: int = 1,
    height: int = 2,
    width: int = 2,
    fill: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return float32 images, masks, and labels of the pack ranks."""
    images = np.full((n, channels, height, width), np.float32(fill), dtype=np.float32)
    masks = np.zeros((n, 1, height, width), dtype=np.float32)
    labels = np.full((n, 1), np.float32(fill), dtype=np.float32)
    return images, masks, labels


def _memory_pack(
    ingest: IngestPath,
    *,
    fill: float,
    splits: SplitIndex,
    n: int,
    norm: NormName = "unit",
    bands: tuple[str, ...] = ("b0",),
    height: int = 2,
    width: int = 2,
    source_doi: str = "10.5281/zenodo.4250706",
) -> ProcessedPack:
    """Return an in-memory pack whose arrays match its meta."""
    images, masks, labels = _arrays(n, channels=len(bands), height=height, width=width, fill=fill)
    provenance = _provenance(ingest, bands=bands, norm=norm)
    meta = dataset_meta_from_provenance(
        provenance,
        dataset_hash="stored-hash",
        source_doi=source_doi,
        n=n,
        height=height,
        width=width,
    )
    return ProcessedPack(images=images, masks=masks, labels=labels, splits=splits, meta=meta)


def test_write_processed_pack_with_group_ids_round_trips(tmp_path: Path) -> None:
    """Group ids become a split, and load returns the provenance fields."""
    dest = tmp_path / "pack"
    group_ids = ["a", "a", "b", "b", "c", "c"]
    images, masks, labels = _arrays(len(group_ids), fill=0.25)
    provenance = _provenance("flight_camera", bands=("red",))
    meta = write_processed_pack(
        dest,
        images,
        masks,
        labels,
        provenance,
        "10.5281/zenodo.4250706",
        group_ids=group_ids,
        recipe=SplitRecipe(seed=1),
    )
    loaded = load_processed_pack(dest)
    assert loaded.meta == meta
    assert loaded.meta.dataset_hash == compute_dataset_hash(dest)
    assert loaded.meta.ingest_path == provenance.ingest_path
    assert loaded.meta.radiometry == provenance.radiometry
    assert loaded.meta.gsd_m == provenance.gsd_m
    assert loaded.meta.extent_m == provenance.extent_m
    assert loaded.meta.weight_table_id == provenance.weight_table_id
    assert loaded.meta.band_names == provenance.band_names
    assert loaded.meta.norm == provenance.norm
    assert loaded.meta.bit_depth == provenance.bit_depth
    assert loaded.meta.in_channels == 1
    np.testing.assert_array_equal(loaded.images, images)
    for group in ("a", "b", "c"):
        rows = [i for i, group_id in enumerate(group_ids) if group_id == group]
        homes = [
            name
            for name in ("train", "val", "test")
            if any(row in loaded.splits.for_name(name) for row in rows)
        ]
        assert len(homes) == 1
        assert all(row in loaded.splits.for_name(homes[0]) for row in rows)


def test_write_processed_pack_uses_precomputed_splits(tmp_path: Path) -> None:
    """A caller-supplied SplitIndex is stored unchanged."""
    dest = tmp_path / "pack"
    images, masks, labels = _arrays(4, fill=0.5)
    index = SplitIndex(train=(0, 1), val=(2,), test=(3,))
    write_processed_pack(
        dest,
        images,
        masks,
        labels,
        _provenance(),
        "doi:synthetic",
        splits=index,
    )
    loaded = load_processed_pack(dest)
    assert loaded.splits == index
    assert loaded.meta.n == 4


def test_write_processed_pack_rejects_bad_inputs(tmp_path: Path) -> None:
    """Channel count, group length, and split source are checked."""
    dest = tmp_path / "pack"
    images, masks, labels = _arrays(3)
    provenance = _provenance(bands=("red", "nir"))
    with pytest.raises(ValueError, match="band_names"):
        write_processed_pack(dest, images, masks, labels, provenance, "doi", splits=_SPLIT3)
    images, masks, labels = _arrays(3)
    with pytest.raises(ValueError, match="group_ids"):
        write_processed_pack(
            dest,
            images,
            masks,
            labels,
            _provenance(),
            "doi",
            group_ids=["a", "b"],
        )
    with pytest.raises(ValueError, match="group_ids or splits"):
        write_processed_pack(dest, images, masks, labels, _provenance(), "doi")
    with pytest.raises(ValueError, match="not both"):
        write_processed_pack(
            dest,
            images,
            masks,
            labels,
            _provenance(),
            "doi",
            group_ids=["a", "b", "c"],
            splits=_SPLIT3,
        )


def test_load_processed_pack_rejects_hash_mismatch(tmp_path: Path) -> None:
    """Changing a hashed file without updating dataset.json raises ValueError."""
    dest = tmp_path / "pack"
    images, masks, labels = _arrays(3, fill=0.1)
    write_processed_pack(dest, images, masks, labels, _provenance(), "doi", splits=_SPLIT3)
    stored = np.load(dest / "labels.npy")
    updated = stored.copy()
    updated[0, 0] = 1.0
    np.save(dest / "labels.npy", updated)
    with pytest.raises(ValueError, match="hash"):
        load_processed_pack(dest)


def test_load_processed_pack_rejects_mask_shape(tmp_path: Path) -> None:
    """A mask tensor that is not (N, 1, H, W) raises ValueError."""
    dest = tmp_path / "pack"
    images, masks, labels = _arrays(3)
    write_processed_pack(dest, images, masks, labels, _provenance(), "doi", splits=_SPLIT3)
    np.save(dest / "masks.npy", np.zeros((3, 2, 2, 2), dtype=np.float32))
    meta = load_dataset_meta(dest / "dataset.json", verify=False)
    write_dataset_meta(
        dest / "dataset.json",
        replace(meta, dataset_hash=compute_dataset_hash(dest)),
    )
    with pytest.raises(ValueError, match="masks"):
        load_processed_pack(dest)


def test_concat_rejects_different_ingest() -> None:
    """Different ingest paths raise ValueError."""
    left = _memory_pack("flight_camera", fill=1.0, splits=_SPLIT3, n=3)
    right = _memory_pack("sentinel2_4250706_study", fill=2.0, splits=_SPLIT3, n=3)
    with pytest.raises(ValueError, match="ingest_path"):
        concat_packs([left, right])
    with pytest.raises(ValueError, match="ingest_path"):
        assert_same_ingest([left, right])


@pytest.mark.parametrize(
    "ingest",
    ["flight_camera", "sentinel2_4250706_study", "sentinel2_4250706_prism_proxy"],
)
def test_concat_same_ingest_sums_n_and_remaps_splits(ingest: IngestPath) -> None:
    """Matching ingest paths stack on N and shift split indices."""
    left_splits = SplitIndex(train=(0,), val=(1,), test=(2,))
    right_splits = SplitIndex(train=(1,), val=(0,), test=(2,))
    left = _memory_pack(ingest, fill=1.0, splits=left_splits, n=3)
    right = _memory_pack(ingest, fill=2.0, splits=right_splits, n=3)
    assert_same_ingest([left, right])
    combined = concat_packs([left, right])
    assert combined.meta.n == 6
    assert combined.meta.dataset_hash == ""
    assert combined.meta.ingest_path == ingest
    assert combined.splits == SplitIndex(train=(0, 4), val=(1, 3), test=(2, 5))
    assert combined.images.shape == (6, 1, 2, 2)
    assert combined.masks.shape == (6, 1, 2, 2)
    assert combined.labels.shape == (6, 1)
    assert float(combined.images[0, 0, 0, 0]) == pytest.approx(1.0)
    assert float(combined.images[3, 0, 0, 0]) == pytest.approx(2.0)
    assert provenance_from_meta(combined.meta) == provenance_from_meta(left.meta)


def test_concat_rejects_layout_mismatch() -> None:
    """Band names, spatial size, and norm must match."""
    base = _memory_pack("flight_camera", fill=1.0, splits=_SPLIT3, n=3)
    other_bands = _memory_pack(
        "flight_camera",
        fill=1.0,
        splits=_SPLIT3,
        n=3,
        bands=("b1",),
    )
    with pytest.raises(ValueError, match="band_names"):
        concat_packs([base, other_bands])
    other_spatial = _memory_pack(
        "flight_camera",
        fill=1.0,
        splits=_SPLIT3,
        n=3,
        height=4,
        width=2,
    )
    with pytest.raises(ValueError, match="spatial"):
        concat_packs([base, other_spatial])
    other_norm = _memory_pack("flight_camera", fill=1.0, splits=_SPLIT3, n=3, norm="band_z")
    with pytest.raises(ValueError, match="norm"):
        concat_packs([base, other_norm])


def test_concat_result_can_be_rewritten(tmp_path: Path) -> None:
    """The in-memory concatenation is a valid input to write_processed_pack."""
    left = _memory_pack("flight_camera", fill=1.0, splits=_SPLIT3, n=3)
    right = _memory_pack("flight_camera", fill=2.0, splits=_SPLIT3, n=3)
    combined = concat_packs([left, right])
    dest = tmp_path / "out"
    assert not dest.exists()
    written = write_processed_pack(
        dest,
        combined.images,
        combined.masks,
        combined.labels,
        provenance_from_meta(combined.meta),
        combined.meta.source_doi,
        splits=combined.splits,
    )
    loaded = load_processed_pack(dest)
    assert loaded.meta.n == 6
    assert loaded.meta.dataset_hash == written.dataset_hash
    assert loaded.splits == combined.splits
    assert loaded.meta.dataset_hash != ""
