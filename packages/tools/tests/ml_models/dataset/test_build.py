"""Tests for the finished-dataset build."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.cli import main
from tools.ml_models.dataset.augment import ELEMENT_NAMES, SHAPE_PRESERVING
from tools.ml_models.dataset.build import build_dataset, build_flight
from tools.ml_models.dataset.manifest import check_compatible, load_manifest
from tools.ml_models.dataset.preprocess import quantize_unit, to_unit
from tools.ml_models.dataset.raw import BinSpec, GsdPair, RawTile, RawTileRef
from tools.ml_models.dataset.sources.flight import FlightTileWrite, write_flight_tile_dir
from tools.ml_models.dataset.spec import BuildSpec
from tools.ml_models.dataset.store import RowRecord, read_images, read_rows


class MemorySource:
    """In-memory raw source for build tests."""

    def __init__(
        self,
        tiles: tuple[RawTile, ...],
        *,
        band_names: tuple[str, ...] = ("BLUE", "GREEN", "RED"),
        extent_m: tuple[float, float] | None = None,
        bins: tuple[BinSpec, ...] = (),
        boom: bool = False,
    ) -> None:
        self.name = "memory"
        self.band_names = band_names
        self.domain = "dn"
        self.bit_depth = 12
        self.source_ref = "test"
        self.extent_m = extent_m
        self.bins = bins
        self._tiles = tiles
        self._boom = boom

    def index(self) -> tuple[RawTileRef, ...]:
        """Return refs in stream order."""
        return tuple(tile.ref for tile in self._tiles)

    def iter_tiles(self) -> Iterator[RawTile]:
        """Yield tiles, optionally failing after the first."""
        for index, tile in enumerate(self._tiles):
            if self._boom and index == 1:
                raise RuntimeError("boom")
            yield tile


class MutatedRefSource(MemorySource):
    """Stream tiles whose ref disagrees with the indexed ref."""

    def iter_tiles(self) -> Iterator[RawTile]:
        """Yield each tile with a flipped label on the streamed ref."""
        for tile in self._tiles:
            yield replace(tile, ref=replace(tile.ref, label=1.0 - tile.ref.label))


def _tile(
    tile_id: str,
    group_id: str,
    gsd: GsdPair,
    shape: tuple[int, int],
    *,
    label: float = 1.0,
    has_mask: bool = False,
    bin_id: str = "",
    fill: int = 100,
) -> RawTile:
    """Build one raw tile of the requested spatial size."""
    height, width = shape
    image = np.full((3, height, width), fill, dtype=np.uint16)
    mask = None
    if has_mask:
        mask = np.zeros((1, height, width), dtype=np.uint8)
        mask[:, :1, :1] = 1
    return RawTile(
        ref=RawTileRef(
            tile_id=tile_id,
            group_id=group_id,
            label=label,
            has_mask=has_mask,
            gsd=gsd,
            frame_id=group_id,
            grid_rc=(0, 0),
            bin_id=bin_id,
        ),
        image=image,
        mask=mask,
    )


def _paired_bins() -> tuple[RawTile, ...]:
    """Three groups, two bins each, one mask per group. Tiles are 4 by 8."""
    gsd = GsdPair(10.0, 20.0)
    shape = (4, 8)
    tiles: list[RawTile] = []
    for group in ("g0", "g1", "g2"):
        for bin_id, masked in (("low", True), ("high", False)):
            tiles.append(
                _tile(
                    f"{group}-{bin_id}",
                    group,
                    gsd,
                    shape,
                    label=1.0 if group == "g0" else 0.0,
                    has_mask=masked,
                    bin_id=bin_id,
                )
            )
    return tuple(tiles)


def _rows_by_split(dest: Path, task: str) -> dict[str, list[RowRecord]]:
    """Collect row records for one task, keyed by split name."""
    found: dict[str, list[RowRecord]] = {"train": [], "val": [], "test": []}
    parent = dest / task
    for split_name in found:
        split_dir = parent / split_name
        if not split_dir.is_dir():
            continue
        for shard in split_dir.iterdir():
            found[split_name].extend(read_rows(shard))
    return found


def test_split_is_shared_by_tasks_and_bins(tmp_path: Path) -> None:
    """One group lands in one split for both tasks and both bins."""
    source = MemorySource(_paired_bins(), extent_m=(80.0, 80.0))
    dest = tmp_path / "ds"
    manifest = build_dataset(source, dest, BuildSpec())
    classifier = _rows_by_split(dest, "classifier")
    segmentor = _rows_by_split(dest, "segmentor")
    homes: dict[str, str] = {}
    for split_name, rows in classifier.items():
        for row in rows:
            homes.setdefault(row.group_id, split_name)
            assert homes[row.group_id] == split_name
    assert set(homes) == {"g0", "g1", "g2"}
    for split_name, rows in segmentor.items():
        for row in rows:
            assert homes[row.group_id] == split_name
    train_bins = {row.bin_id for row in classifier["train"]}
    assert train_bins == {"low", "high"}
    assert manifest.shards
    assert any(shard.n_positive > 0 for shard in manifest.shards)


def test_train_augmentation_is_four_elements_when_nonsquare(tmp_path: Path) -> None:
    """Val and test stay unaugmented. A 4 by 8 train tile keeps four elements."""
    source = MemorySource(_paired_bins(), extent_m=(80.0, 80.0))
    dest = tmp_path / "ds"
    build_dataset(source, dest, BuildSpec())
    rows = _rows_by_split(dest, "classifier")
    assert {row.element for row in rows["val"]} == {"id"}
    assert {row.element for row in rows["test"]} == {"id"}
    assert {row.element for row in rows["train"]} == set(SHAPE_PRESERVING)
    train_ids = {row.tile_id for row in rows["train"]}
    assert len(rows["train"]) == len(train_ids) * len(SHAPE_PRESERVING)


def test_train_augmentation_is_eight_elements_when_square(tmp_path: Path) -> None:
    """A square train tile keeps all eight dihedral elements."""
    gsd = GsdPair(10.0, 10.0)
    tiles = tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8), has_mask=True) for index in range(3))
    dest = tmp_path / "ds"
    build_dataset(MemorySource(tiles, extent_m=(80.0, 80.0)), dest, BuildSpec())
    rows = _rows_by_split(dest, "classifier")
    assert {row.element for row in rows["train"]} == set(ELEMENT_NAMES)
    train_ids = {row.tile_id for row in rows["train"]}
    assert len(rows["train"]) == len(train_ids) * len(ELEMENT_NAMES)


def test_hash_is_deterministic(tmp_path: Path) -> None:
    """Two builds of the same source share a content hash."""
    tiles = _paired_bins()
    first = build_dataset(MemorySource(tiles, extent_m=(80.0, 80.0)), tmp_path / "a", BuildSpec())
    second = build_dataset(MemorySource(tiles, extent_m=(80.0, 80.0)), tmp_path / "b", BuildSpec())
    assert first.dataset_hash == second.dataset_hash
    assert load_manifest(tmp_path / "a" / "dataset.json").dataset_hash == first.dataset_hash


def test_failure_leaves_no_destination(tmp_path: Path) -> None:
    """A stream error removes the partial directory and does not create dest."""
    dest = tmp_path / "ds"
    source = MemorySource(_paired_bins(), extent_m=(80.0, 80.0), boom=True)
    with pytest.raises(RuntimeError, match="boom"):
        build_dataset(source, dest, BuildSpec())
    assert not dest.exists()
    assert not list(tmp_path.glob(".ds.partial-*"))


def test_failed_build_preserves_other_partial_directories(tmp_path: Path) -> None:
    """A failed build removes only its own temporary directory."""
    other = tmp_path / ".ds.partial-other"
    other.mkdir()
    marker = other / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    dest = tmp_path / "ds"
    source = MemorySource(_paired_bins(), extent_m=(80.0, 80.0), boom=True)
    with pytest.raises(RuntimeError, match="boom"):
        build_dataset(source, dest, BuildSpec())
    assert not dest.exists()
    assert marker.read_text(encoding="utf-8") == "keep"
    assert set(tmp_path.glob(".ds.partial-*")) == {other}


def test_nonbinary_label_is_rejected(tmp_path: Path) -> None:
    """Labels must be 0.0 or 1.0."""
    tiles = tuple(
        _tile(f"t{index}", f"g{index}", GsdPair(10.0, 10.0), (8, 8), label=0.5)
        for index in range(3)
    )
    source = MemorySource(tiles, extent_m=(80.0, 80.0))
    with pytest.raises(ValueError, match="binary"):
        build_dataset(source, tmp_path / "ds", BuildSpec())


def test_nonfinite_extent_is_rejected(tmp_path: Path) -> None:
    """A non-finite source extent fails the size rule."""
    tiles = tuple(
        _tile(f"t{index}", f"g{index}", GsdPair(10.0, 10.0), (8, 8)) for index in range(3)
    )
    source = MemorySource(tiles, extent_m=(float("nan"), 80.0))
    with pytest.raises(ValueError, match="extent_m"):
        build_dataset(source, tmp_path / "ds", BuildSpec())


def test_nonbinary_mask_is_rejected(tmp_path: Path) -> None:
    """Mask pixels must be 0 or 1."""
    tiles = []
    for index in range(3):
        tile = _tile(f"t{index}", f"g{index}", GsdPair(10.0, 10.0), (8, 8), has_mask=True)
        assert tile.mask is not None
        tile.mask[0, 0, 0] = 2
        tiles.append(tile)
    source = MemorySource(tuple(tiles), extent_m=(80.0, 80.0))
    with pytest.raises(ValueError, match="binary pixels"):
        build_dataset(source, tmp_path / "ds", BuildSpec())


def test_nonfinite_image_is_rejected(tmp_path: Path) -> None:
    """Streamed pixels must be finite."""
    gsd = GsdPair(10.0, 10.0)
    bad = replace(
        _tile("t0", "g0", gsd, (8, 8)),
        image=np.full((3, 8, 8), np.nan, dtype=np.float32),
    )
    rest = tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8)) for index in range(1, 3))
    source = MemorySource((bad, *rest), extent_m=(80.0, 80.0))
    with pytest.raises(ValueError, match="non-finite"):
        build_dataset(source, tmp_path / "ds", BuildSpec())


def test_stream_ref_must_match_index(tmp_path: Path) -> None:
    """A streamed row whose ref differs from its index row is rejected."""
    source = MutatedRefSource(_paired_bins(), extent_m=(80.0, 80.0))
    with pytest.raises(ValueError, match="order mismatch"):
        build_dataset(source, tmp_path / "ds", BuildSpec())


def test_anisotropic_square_tile_keeps_axis_preserving_elements(tmp_path: Path) -> None:
    """Anisotropic stored GSD drops axis-swapping elements even on square tiles."""
    gsd = GsdPair(10.0, 20.0)
    tiles = tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8), has_mask=True) for index in range(3))
    dest = tmp_path / "ds"
    build_dataset(MemorySource(tiles, extent_m=(80.0, 160.0)), dest, BuildSpec())
    rows = _rows_by_split(dest, "classifier")
    assert {row.element for row in rows["train"]} == set(SHAPE_PRESERVING)


def test_existing_destination_is_unchanged(tmp_path: Path) -> None:
    """A second build refuses to replace a finished dataset."""
    tiles = _paired_bins()
    dest = tmp_path / "ds"
    first = build_dataset(MemorySource(tiles, extent_m=(80.0, 80.0)), dest, BuildSpec())
    with pytest.raises(FileExistsError):
        build_dataset(MemorySource(tiles, extent_m=(80.0, 80.0)), dest, BuildSpec())
    assert load_manifest(dest / "dataset.json").dataset_hash == first.dataset_hash


def test_band_mismatch_is_rejected(tmp_path: Path) -> None:
    """Source bands must equal the spec's input bands."""
    tiles = tuple(
        _tile(f"t{index}", f"g{index}", GsdPair(10.0, 10.0), (8, 8)) for index in range(3)
    )
    source = MemorySource(tiles, band_names=("NIR", "RED", "GREEN"), extent_m=(80.0, 80.0))
    with pytest.raises(ValueError, match="band_names"):
        build_dataset(source, tmp_path / "ds", BuildSpec())


def test_nonpositive_gsd_is_rejected(tmp_path: Path) -> None:
    """A zero GSD component is rejected."""
    tiles = tuple(_tile(f"t{index}", f"g{index}", GsdPair(0.0, 10.0), (8, 8)) for index in range(3))
    with pytest.raises(ValueError, match="gsd"):
        build_dataset(MemorySource(tiles, extent_m=(80.0, 80.0)), tmp_path / "ds", BuildSpec())


def test_inconsistent_size_is_rejected(tmp_path: Path) -> None:
    """Pixel size must match round(extent / gsd)."""
    gsd = GsdPair(10.0, 10.0)
    tiles = tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8)) for index in range(3))
    source = MemorySource(tiles, extent_m=(100.0, 100.0))
    with pytest.raises(ValueError, match="inconsistent"):
        build_dataset(source, tmp_path / "ds", BuildSpec())


def test_uint16_round_trip(tmp_path: Path) -> None:
    """Stored pixels equal round(unit * 65535) for the identity element."""
    gsd = GsdPair(10.0, 10.0)
    tiles = tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8), fill=4095) for index in range(3))
    dest = tmp_path / "ds"
    build_dataset(MemorySource(tiles, extent_m=(80.0, 80.0)), dest, BuildSpec())
    rows = _rows_by_split(dest, "classifier")
    identity = next(
        row for row in rows["train"] + rows["val"] + rows["test"] if row.element == "id"
    )
    shard = dest / "classifier"
    stored = None
    for split_name in ("train", "val", "test"):
        directory = shard / split_name / "8x8"
        if not directory.is_dir():
            continue
        shard_rows = read_rows(directory)
        for index, row in enumerate(shard_rows):
            if row.tile_id == identity.tile_id and row.element == "id":
                stored = read_images(directory)[index]
    assert stored is not None
    expected = quantize_unit(to_unit(tiles[0].image, "dn", 12))
    np.testing.assert_array_equal(stored, expected)


def test_check_compatible_requires_shared_reference_and_bands(tmp_path: Path) -> None:
    """Datasets must share bands, unit norm, and the GSD reference."""
    gsd = GsdPair(10.0, 10.0)
    tiles = tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8)) for index in range(3))
    base = build_dataset(
        MemorySource(tiles, extent_m=(80.0, 80.0)),
        tmp_path / "base",
        BuildSpec(),
    )
    other = build_dataset(
        MemorySource(tiles, extent_m=(80.0, 80.0)),
        tmp_path / "other",
        BuildSpec(gsd_reference_m=20.0),
    )
    alt_tiles = tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8)) for index in range(3))
    alt = build_dataset(
        MemorySource(alt_tiles, band_names=("A", "B", "C"), extent_m=(80.0, 80.0)),
        tmp_path / "alt",
        BuildSpec(input_bands=("A", "B", "C")),
    )
    check_compatible([base, base])
    with pytest.raises(ValueError, match="gsd_reference_m"):
        check_compatible([base, other])
    with pytest.raises(ValueError, match="band_names"):
        check_compatible([base, alt])


def test_cli_reports_bad_dataset_args(tmp_path: Path) -> None:
    """Bad dataset arguments surface as a parameter error, not a traceback."""
    missing_dir = main(["dataset", "build", "--source", "flight", "--out", str(tmp_path / "ds")])
    assert missing_dir != 0
    removed_source = main(
        ["dataset", "build", "--source", "synthetic", "--out", str(tmp_path / "ds3")]
    )
    assert removed_source != 0
    missing_source = main(
        [
            "dataset",
            "build",
            "--source",
            "flight",
            "--source-dir",
            str(tmp_path / "absent"),
            "--out",
            str(tmp_path / "ds2"),
        ]
    )
    assert missing_source != 0


def _flight_tile(index: int, **overrides: object) -> FlightTileWrite:
    """One flight tile write for ``build_flight`` tests."""
    fields: dict[str, object] = {
        "tile_id": f"f{index // 8}-{index % 8}-{index}",
        "frame_id": f"f{index // 8}",
        "row": index % 8,
        "col": index % 8,
        "label": 1.0,
        "theta_g_deg": 15.0,
        "gsd": GsdPair(15.87, 15.87),
        "image": np.full((3, 193, 258), 10, dtype=np.uint16),
    }
    fields.update(overrides)
    return FlightTileWrite(**fields)  # type: ignore[arg-type]


def test_build_flight_rejects_nominal_rows(tmp_path: Path) -> None:
    """The standard flight build refuses nominal-GSD captures."""
    source_dir = tmp_path / "flight-src"
    write_flight_tile_dir(source_dir, [_flight_tile(0), _flight_tile(1, gsd_nominal=True)])
    with pytest.raises(ValueError, match="nominal"):
        build_flight(source_dir, tmp_path / "ds")


def _research_tile(tile_id: str, group_id: str, **ref_overrides: object) -> RawTile:
    """One 4x8 research-source tile with overridable ref fields."""
    fields: dict[str, object] = {
        "tile_id": tile_id,
        "group_id": group_id,
        "label": 1.0,
        "has_mask": False,
        "gsd": GsdPair(10.0, 20.0),
        "frame_id": group_id,
        "grid_rc": (0, 0),
        "bin_id": "",
    }
    fields.update(ref_overrides)
    return RawTile(
        ref=RawTileRef(**fields),  # type: ignore[arg-type]
        image=np.full((3, 4, 8), 5, dtype=np.uint16),
        mask=None,
    )


def test_research_source_records_nominal_flag(tmp_path: Path) -> None:
    """A custom research source keeps gsd_nominal and theta on the row."""
    tiles = tuple(_research_tile(f"t{index}", f"g{index}") for index in range(3)) + (
        _research_tile("t-nominal", "g0", theta_g_deg=30.0, gsd_nominal=True),
    )
    source = MemorySource(tiles, extent_m=(80.0, 80.0))
    dest = tmp_path / "ds"
    build_dataset(source, dest, BuildSpec())
    rows = [
        row
        for rows in _rows_by_split(dest, "classifier").values()
        for row in rows
        if row.tile_id == "t-nominal" and row.element == "id"
    ]
    assert len(rows) == 1
    assert rows[0].gsd_nominal is True
    assert rows[0].theta_g_deg == 30.0


def test_build_rejects_non_finite_theta(tmp_path: Path) -> None:
    """A non-finite theta_g_deg fails the build."""
    tiles = tuple(_research_tile(f"t{index}", f"g{index}") for index in range(3)) + (
        _research_tile("t-bad", "g0", theta_g_deg=float("nan")),
    )
    source = MemorySource(tiles, extent_m=(80.0, 80.0))
    with pytest.raises(ValueError, match="theta_g_deg"):
        build_dataset(source, tmp_path / "ds", BuildSpec())
