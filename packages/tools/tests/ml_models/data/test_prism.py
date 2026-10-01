"""Prism mix and 76 px proxy chips. No archives and no network."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.data.bands import ZENODO_BAND_IDS
from tools.ml_models.data.meta import NormName, Provenance, dataset_meta_from_provenance
from tools.ml_models.data.pack import ProcessedPack, concat_packs, load_processed_pack
from tools.ml_models.data.prism import (
    CHIP_HW,
    FRAME_HW,
    GSD_MATCH_TOLERANCE,
    PROXY_GSD_M,
    PROXY_SIDE_PX,
    TILE_GRID,
    TILE_HW,
    WeightTable,
    _feather_paste,
    load_weight_table,
    mix_prism,
    to_proxy_chip,
    union_location_split,
    write_prism_pack,
    write_tile_pack,
)
from tools.ml_models.data.split import SplitIndex
from tools.ml_models.data.zenodo import TileIndex, TileRef, to_native_stack

_ROOT = Path(__file__).resolve().parents[5]
_WEIGHTS = _ROOT / "data" / "manifests" / "ap3200t_s2_weights.toml"
_POLYGON = (np.array([[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]], dtype=np.float32),)


def _stack_at(band_id: str, value: float, height: int, width: int) -> np.ndarray:
    """Return a 13-band stack that is ``value`` on one band and 0 elsewhere."""
    stack = np.zeros((len(ZENODO_BAND_IDS), height, width), dtype=np.float32)
    stack[ZENODO_BAND_IDS.index(band_id)] = value
    return stack


def test_mix_prism_b1_only_is_blue() -> None:
    """Counts of 10000 in B1 become blue 0.59 and leave green and red at 0."""
    table = load_weight_table(_WEIGHTS)
    assert table.id == "ap3200t_s2_figure"
    mixed = mix_prism(_stack_at("B1", 10000.0, 4, 5), ZENODO_BAND_IDS, table)
    assert mixed.shape == (3, 4, 5)
    assert mixed.dtype == np.float32
    assert mixed[0, 0, 0] == pytest.approx(0.59)
    assert mixed[1, 0, 0] == pytest.approx(0.0)
    assert mixed[2, 0, 0] == pytest.approx(0.0)
    assert np.allclose(mixed[0], 0.59)
    assert np.allclose(mixed[1], 0.0)
    assert np.allclose(mixed[2], 0.0)


def test_missing_weight_table_raises(tmp_path: Path) -> None:
    """A missing weight table is a FileNotFoundError."""
    missing = tmp_path / "absent.toml"
    with pytest.raises(FileNotFoundError):
        load_weight_table(missing)


def test_weight_sum_must_be_one(tmp_path: Path) -> None:
    """A color whose weights do not sum to 1 is refused."""
    path = tmp_path / "bad.toml"
    path.write_text(
        'id = "bad"\n\n[blue]\nB1 = 0.5\n\n[green]\nB3 = 1.0\n\n[red]\nB4 = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="blue"):
        load_weight_table(path)


def test_unknown_band_id_raises() -> None:
    """A table band that is absent from the stack ids is refused."""
    table = WeightTable(id="x", blue={"NOPE": 1.0}, green={"B3": 1.0}, red={"B4": 1.0})
    stack = np.zeros((len(ZENODO_BAND_IDS), 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="NOPE"):
        mix_prism(stack, ZENODO_BAND_IDS, table)


def test_to_proxy_chip_shape() -> None:
    """A 13 by 120 by 120 stack becomes a 76 px image and mask."""
    table = load_weight_table(_WEIGHTS)
    image, mask = to_proxy_chip(
        _stack_at("B1", 10000.0, 120, 120),
        ZENODO_BAND_IDS,
        table,
        _POLYGON,
    )
    assert image.shape == (3, PROXY_SIDE_PX, PROXY_SIDE_PX)
    assert mask.shape == (1, PROXY_SIDE_PX, PROXY_SIDE_PX)
    assert image.dtype == np.float32
    assert mask.dtype == np.float32
    assert image.shape[1] == 76
    assert np.allclose(image[0], 0.59)
    assert np.allclose(image[1], 0.0)
    assert np.allclose(image[2], 0.0)
    assert mask.max() == 1.0
    assert mask.min() == 0.0


def test_script_refuses_a_missing_weight_table(tmp_path: Path) -> None:
    """The CLI raises before it needs an image archive."""
    script = _ROOT / "scripts" / "preprocess_prism_proxy.py"
    spec = importlib.util.spec_from_file_location("preprocess_prism_proxy", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = tmp_path / "no-weights.toml"
    with pytest.raises(FileNotFoundError, match="no-weights.toml"):
        module.main(
            [
                "--images",
                str(tmp_path / "images.tar"),
                "--labels",
                str(tmp_path / "labels.tar"),
                "--weights",
                str(missing),
                "--dest",
                str(tmp_path / "pack"),
            ]
        )
    assert not (tmp_path / "images.tar").exists()


def _tile(location: str, polygons: tuple[np.ndarray, ...] | None) -> TileRef:
    """Return one tile whose stem starts with ``location``."""
    stem = f"{location}_2019-01-01T00:00:00.000Z_0"
    return TileRef(
        stem=stem,
        location_id=location,
        positive=True,
        member_name=f"positive/{stem}.tif",
        polygons=polygons,
    )


def test_write_prism_pack_keeps_annotated_negatives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty polygon tuple stays in the pack. A missing annotation does not."""
    polygon = (
        np.array([[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]], dtype=np.float32),
    )
    tiles = (
        _tile("10", polygon),
        _tile("20", ()),
        _tile("30", polygon),
        _tile("40", polygon),
        _tile("50", None),
    )

    def _index(_images: Path, _labels: Path) -> TileIndex:
        return TileIndex(tiles=tiles)

    def _stacks(
        _images: Path,
        selected: Sequence[TileRef],
    ) -> Iterator[tuple[TileRef, np.ndarray, tuple[str, ...]]]:
        stack = np.zeros((len(ZENODO_BAND_IDS), 120, 120), dtype=np.float32)
        stack[0] = 10000.0
        descriptions = tuple("" for _band in ZENODO_BAND_IDS)
        for tile in selected:
            yield tile, stack, descriptions

    monkeypatch.setattr("tools.ml_models.data.prism.build_index", _index)
    monkeypatch.setattr("tools.ml_models.data.prism.iter_stacks", _stacks)
    dest = tmp_path / "pack"
    write_prism_pack(tmp_path / "images.tar", tmp_path / "labels.tar", _WEIGHTS, dest)
    pack = load_processed_pack(dest)
    assert pack.labels.shape == (4, 1)
    assert pack.labels.ravel().tolist() == [1.0, 0.0, 1.0, 1.0]
    assert float(pack.masks[1].max()) == 0.0
    assert float(pack.masks[0].max()) == 1.0


@pytest.mark.parametrize(("height", "width"), [(120, 119), (119, 120), (121, 120), (120, 121)])
def test_near_native_stack_is_fitted_before_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    height: int,
    width: int,
) -> None:
    """A 119 or 121 side is padded or cropped to 120 before the 76 px resample."""
    stack = np.zeros((len(ZENODO_BAND_IDS), height, width), dtype=np.float32)
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    stack[0] = rows + cols
    stack[0, -1, :] = 9000.0
    stack[0, :, -1] = 9000.0
    tiles = (
        _tile("10", _POLYGON),
        _tile("20", _POLYGON),
        _tile("30", ()),
    )

    def _index(_images: Path, _labels: Path) -> TileIndex:
        return TileIndex(tiles=tiles)

    def _stacks(
        _images: Path,
        selected: Sequence[TileRef],
    ) -> Iterator[tuple[TileRef, np.ndarray, tuple[str, ...]]]:
        descriptions = tuple("" for _band in ZENODO_BAND_IDS)
        for tile in selected:
            yield tile, stack, descriptions

    monkeypatch.setattr("tools.ml_models.data.prism.build_index", _index)
    monkeypatch.setattr("tools.ml_models.data.prism.iter_stacks", _stacks)
    dest = tmp_path / "pack"
    write_prism_pack(tmp_path / "images.tar", tmp_path / "labels.tar", _WEIGHTS, dest)
    pack = load_processed_pack(dest)
    table = load_weight_table(_WEIGHTS)
    fitted, _mask = to_proxy_chip(to_native_stack(stack), ZENODO_BAND_IDS, table, _POLYGON)
    raw, _raw_mask = to_proxy_chip(stack, ZENODO_BAND_IDS, table, _POLYGON)
    stored = np.clip(fitted, 0.0, 1.0).astype(np.float32)
    stretched = np.clip(raw, 0.0, 1.0).astype(np.float32)
    assert pack.images.shape == (3, 3, PROXY_SIDE_PX, PROXY_SIDE_PX)
    assert np.allclose(pack.images[0], stored)
    assert np.allclose(pack.images[1], stored)
    assert not np.allclose(stored, stretched)


def test_tile_grid_covers_the_flight_frame() -> None:
    """Eight by eight tiles of 193 by 258 equal the 1544 by 2064 frame."""
    assert CHIP_HW == (76, 76)
    assert TILE_HW == (193, 258)
    assert FRAME_HW == (1544, 2064)
    assert TILE_GRID == (8, 8)
    assert TILE_GRID[0] * TILE_HW[0] == FRAME_HW[0]
    assert TILE_GRID[1] * TILE_HW[1] == FRAME_HW[1]
    assert GSD_MATCH_TOLERANCE == pytest.approx(0.01)
    assert PROXY_GSD_M == pytest.approx(1200 / 76)


def _chip_pack(
    *,
    fills: tuple[float, ...],
    labels: tuple[float, ...],
    splits: SplitIndex,
    group_ids: tuple[str, ...],
    gsd_m: float = PROXY_GSD_M,
    height: int = 76,
    width: int = 76,
) -> ProcessedPack:
    """Return an in-memory 3-band chip pack."""
    n = len(group_ids)
    images = np.zeros((n, 3, height, width), dtype=np.float32)
    masks = np.zeros((n, 1, height, width), dtype=np.float32)
    label_array = np.zeros((n, 1), dtype=np.float32)
    for row, (fill, label) in enumerate(zip(fills, labels, strict=True)):
        images[row] = np.float32(fill)
        label_array[row, 0] = np.float32(label)
        if label > 0.0:
            masks[row, 0, 30:40, 30:40] = 1.0
    provenance = Provenance(
        ingest_path="sentinel2_4250706_prism_proxy",
        radiometry="s2_l2a_reflectance",
        gsd_m=gsd_m,
        extent_m=1200.0,
        weight_table_id="ap3200t_s2_figure",
        band_names=("BLUE", "GREEN", "RED"),
        norm="unit",
        bit_depth=12,
    )
    meta = dataset_meta_from_provenance(
        provenance,
        dataset_hash="stored",
        source_doi="10.5281/zenodo.4250706",
        n=n,
        height=height,
        width=width,
    )
    return ProcessedPack(
        images=images,
        masks=masks,
        labels=label_array,
        splits=splits,
        meta=meta,
        group_ids=group_ids,
    )


def _five_chip_pack() -> ProcessedPack:
    """Return two splits with a positive and a negative, plus one test negative."""
    return _chip_pack(
        fills=(1.0, 0.0, 0.5, 1.0, 0.25),
        labels=(1.0, 0.0, 0.0, 1.0, 0.0),
        splits=SplitIndex(train=(0, 1), val=(2, 3), test=(4,)),
        group_ids=("loc-a", "loc-b", "loc-c", "loc-d", "loc-e"),
    )


def test_write_tile_pack_stores_same_split_tiles(tmp_path: Path) -> None:
    """Each chip becomes a 193 by 258 tile at the chip ground sample distance."""
    source = _five_chip_pack()
    dest = tmp_path / "tiles"
    meta = write_tile_pack(source, dest, seed=0)
    pack = load_processed_pack(dest)
    assert pack.meta == meta
    assert pack.images.shape == (5, 3, 193, 258)
    assert pack.masks.shape == (5, 1, 193, 258)
    assert pack.labels.shape == (5, 1)
    assert pack.images.dtype == np.float32
    assert pack.masks.dtype == np.float32
    assert pack.labels.dtype == np.float32
    assert pack.group_ids == source.group_ids
    assert pack.meta.ingest_path == "sentinel2_4250706_prism_tile"
    assert pack.meta.ingest_path != source.meta.ingest_path
    assert pack.meta.gsd_m == PROXY_GSD_M
    assert pack.meta.band_names == ("BLUE", "GREEN", "RED")
    assert pack.meta.norm == "unit"
    assert pack.meta.radiometry == "s2_l2a_reflectance"
    assert pack.meta.extent_m == 1200.0
    assert float(pack.masks[0].max()) == 1.0
    assert float(pack.labels[0, 0]) == 1.0
    assert float(pack.images[1].max()) == 0.0
    assert float(pack.masks[1].max()) == 0.0
    assert float(pack.labels[1, 0]) == 0.0
    assert np.allclose(pack.images[2], 0.5)
    assert float(pack.masks[2].max()) == 0.0
    assert float(pack.images[3].min()) >= 0.5 - 1e-6
    assert float(pack.masks[3].max()) == 1.0
    assert float(pack.labels[3, 0]) == 1.0
    assert np.allclose(pack.images[4], 0.25)
    assert float(pack.masks[4].max()) == 0.0
    assert float(pack.images.max()) <= 1.0
    assert float(pack.images.min()) >= 0.0


def test_write_tile_pack_seed_changes_the_positive_offset(tmp_path: Path) -> None:
    """The same seed repeats a tile. A new seed moves the pasted plume."""
    source = _five_chip_pack()
    first = tmp_path / "a"
    second = tmp_path / "b"
    third = tmp_path / "c"
    write_tile_pack(source, first, seed=0)
    write_tile_pack(source, second, seed=0)
    write_tile_pack(source, third, seed=1)
    again = load_processed_pack(second)
    other = load_processed_pack(third)
    original = load_processed_pack(first)
    np.testing.assert_array_equal(original.images, again.images)
    np.testing.assert_array_equal(original.masks, again.masks)
    assert not np.array_equal(original.masks[0], other.masks[0])


def test_feather_paste_blends_the_border() -> None:
    """Interior pixels copy the chip. Border pixels blend with the frame."""
    frame = np.zeros((1, 6, 6), dtype=np.float32)
    chip = np.ones((1, 4, 4), dtype=np.float32)
    _feather_paste(frame, chip, 1, 1, 1)
    assert frame[0, 1, 1] == pytest.approx(0.5)
    assert frame[0, 2, 2] == pytest.approx(1.0)
    assert frame[0, 0, 0] == 0.0
    assert frame[0, 4, 4] == pytest.approx(0.5)


def test_write_tile_pack_rejects_a_split_without_negatives(tmp_path: Path) -> None:
    """A positive chip needs a same-split negative mosaic."""
    source = _chip_pack(
        fills=(1.0, 1.0, 1.0),
        labels=(1.0, 1.0, 1.0),
        splits=SplitIndex(train=(0,), val=(1,), test=(2,)),
        group_ids=("loc-a", "loc-b", "loc-c"),
    )
    with pytest.raises(ValueError, match="negative"):
        write_tile_pack(source, tmp_path / "tiles", seed=0)


def test_write_tile_pack_rejects_a_distant_gsd(tmp_path: Path) -> None:
    """A chip pack whose ground sample distance is 2 percent off is refused."""
    source = _chip_pack(
        fills=(1.0, 0.0, 0.0),
        labels=(1.0, 0.0, 0.0),
        splits=SplitIndex(train=(0, 1, 2), val=(), test=()),
        group_ids=("loc-a", "loc-b", "loc-c"),
        gsd_m=PROXY_GSD_M * 1.02,
    )
    with pytest.raises(ValueError, match="gsd"):
        write_tile_pack(source, tmp_path / "tiles", seed=0)


def _union_pack(
    group_ids: tuple[str, ...],
    *,
    height: int,
    width: int,
    gsd_m: float,
    bands: tuple[str, ...] = ("BLUE", "GREEN", "RED"),
    norm: NormName = "unit",
) -> ProcessedPack:
    """Return a pack whose stored split is not the union assignment."""
    n = len(group_ids)
    images = np.zeros((n, len(bands), height, width), dtype=np.float32)
    masks = np.zeros((n, 1, height, width), dtype=np.float32)
    labels = np.zeros((n, 1), dtype=np.float32)
    band_mean = tuple(0.0 for _band in bands) if norm == "band_z" else ()
    band_std = tuple(1.0 for _band in bands) if norm == "band_z" else ()
    provenance = Provenance(
        ingest_path="flight_camera",
        radiometry="s2_l2a_reflectance",
        gsd_m=gsd_m,
        extent_m=20.0,
        weight_table_id="table-a",
        band_names=bands,
        norm=norm,
        bit_depth=12,
        band_mean=band_mean,
        band_std=band_std,
    )
    meta = dataset_meta_from_provenance(
        provenance,
        dataset_hash="stored",
        source_doi="10.5281/zenodo.4250706",
        n=n,
        height=height,
        width=width,
    )
    return ProcessedPack(
        images=images,
        masks=masks,
        labels=labels,
        splits=SplitIndex(train=tuple(range(n)), val=(), test=()),
        meta=meta,
        group_ids=group_ids,
    )


def test_union_location_split_shares_one_site_across_sizes() -> None:
    """Overlapping locations share a split when spatial sizes differ."""
    left = _union_pack(("a", "b", "c", "a"), height=4, width=4, gsd_m=10.0)
    right = _union_pack(("c", "d", "e"), height=9, width=6, gsd_m=10.05)
    with pytest.raises(ValueError):
        concat_packs([left, right])
    result = union_location_split(left, right)
    assert result.groups == ("a", "b", "c", "d", "e")
    names = dict(zip(result.groups, result.names, strict=True))
    assert names["c"] in {"train", "val", "test"}
    left_home = {
        name
        for name in ("train", "val", "test")
        if any(row in result.left.for_name(name) for row in (0, 3))
    }
    right_home = {name for name in ("train", "val", "test") if 0 in result.right.for_name(name)}
    assert left_home == {names["a"]}
    assert right_home == {names["c"]}
    assert "val" in result.names
    assert "test" in result.names
    covered = result.left.train + result.left.val + result.left.test
    assert sorted(covered) == [0, 1, 2, 3]


def test_union_location_split_rejects_bands_norm_or_gsd() -> None:
    """Band names, norm, and a 1 percent ground-sample-distance gap are required."""
    base = _union_pack(("a", "b", "c"), height=2, width=2, gsd_m=10.0)
    other_bands = _union_pack(
        ("a", "b", "d"),
        height=2,
        width=2,
        gsd_m=10.0,
        bands=("RED", "GREEN", "BLUE"),
    )
    other_norm = _union_pack(("a", "b", "d"), height=2, width=2, gsd_m=10.0, norm="band_z")
    other_gsd = _union_pack(("a", "b", "d"), height=2, width=2, gsd_m=10.2)
    with pytest.raises(ValueError, match="band_names"):
        union_location_split(base, other_bands)
    with pytest.raises(ValueError, match="norm"):
        union_location_split(base, other_norm)
    with pytest.raises(ValueError, match="gsd"):
        union_location_split(base, other_gsd)
