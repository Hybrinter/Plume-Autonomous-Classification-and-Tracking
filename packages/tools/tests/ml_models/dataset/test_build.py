"""Tests for the finished-dataset build."""

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from tools.ml_models.cli import main
from tools.ml_models.dataset.augment import ELEMENT_NAMES, SHAPE_PRESERVING, apply_dihedral
from tools.ml_models.dataset.build import build_dataset, build_flight
from tools.ml_models.dataset.loader import ShardDataset
from tools.ml_models.dataset.manifest import compute_dataset_hash, load_manifest
from tools.ml_models.dataset.raw import (
    BinSpec,
    ConditionTag,
    GsdPair,
    ObservationMetadata,
    RawTile,
    RawTileRef,
)
from tools.ml_models.dataset.sources.flight import FlightTileWrite, write_flight_tile_dir
from tools.ml_models.dataset.spec import BuildSpec
from tools.ml_models.dataset.split import SplitRecipe, assign_group_splits
from tools.ml_models.dataset.store import (
    RowRecord,
    read_gsd,
    read_images,
    read_labels,
    read_masks,
    read_rows,
)


class MemorySource:
    """In-memory raw source for build tests."""

    def __init__(
        self,
        tiles: tuple[RawTile, ...],
        *,
        band_names: tuple[str, ...] = ("BLUE", "GREEN", "RED"),
        domain: str = "unit",
        bins: tuple[BinSpec, ...] = (),
        boom: bool = False,
    ) -> None:
        self.name = "memory"
        self.band_names = band_names
        self.domain = domain
        self.source_ref = "test"
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
    fill: float = 0.5,
    channels: int = 3,
) -> RawTile:
    """Build one raw tile of the requested spatial size."""
    height, width = shape
    image = np.full((channels, height, width), fill, dtype=np.float32)
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
            height=height,
            width=width,
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


def _shard_dirs(dest: Path, task: str) -> list[Path]:
    """Return every shard directory for one task across all splits."""
    found: list[Path] = []
    for split_name in ("train", "val", "test"):
        split_dir = dest / task / split_name
        if split_dir.is_dir():
            found.extend(sorted(split_dir.iterdir()))
    return found


def test_split_is_shared_by_tasks_and_bins(tmp_path: Path) -> None:
    """One group lands in one split for both tasks and both bins."""
    source = MemorySource(_paired_bins())
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
    source = MemorySource(_paired_bins())
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
    build_dataset(MemorySource(tiles), dest, BuildSpec())
    rows = _rows_by_split(dest, "classifier")
    assert {row.element for row in rows["train"]} == set(ELEMENT_NAMES)
    train_ids = {row.tile_id for row in rows["train"]}
    assert len(rows["train"]) == len(train_ids) * len(ELEMENT_NAMES)


def test_hash_is_deterministic(tmp_path: Path) -> None:
    """Two builds of the same source share a content hash."""
    tiles = _paired_bins()
    first = build_dataset(MemorySource(tiles), tmp_path / "a", BuildSpec())
    second = build_dataset(MemorySource(tiles), tmp_path / "b", BuildSpec())
    assert first.dataset_hash == second.dataset_hash
    assert load_manifest(tmp_path / "a" / "dataset.json").dataset_hash == first.dataset_hash


def test_failure_leaves_no_destination(tmp_path: Path) -> None:
    """A stream error removes the partial directory and does not create dest."""
    dest = tmp_path / "ds"
    source = MemorySource(_paired_bins(), boom=True)
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
    source = MemorySource(_paired_bins(), boom=True)
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
    source = MemorySource(tiles)
    with pytest.raises(ValueError, match="binary"):
        build_dataset(source, tmp_path / "ds", BuildSpec())


def test_invalid_indexed_dimensions_are_rejected(tmp_path: Path) -> None:
    """Indexed height and width must be positive integers."""
    gsd = GsdPair(10.0, 10.0)
    tiles = tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8)) for index in range(3))
    bad = replace(tiles[0], ref=replace(tiles[0].ref, height=0))
    with pytest.raises(ValueError, match="indexed height"):
        build_dataset(MemorySource((bad, *tiles[1:])), tmp_path / "ds", BuildSpec())
    bad = replace(tiles[0], ref=replace(tiles[0].ref, width=-2))
    with pytest.raises(ValueError, match="indexed width"):
        build_dataset(MemorySource((bad, *tiles[1:])), tmp_path / "ds2", BuildSpec())


def test_nonbinary_mask_is_rejected(tmp_path: Path) -> None:
    """Mask pixels must be 0 or 1."""
    tiles = []
    for index in range(3):
        tile = _tile(f"t{index}", f"g{index}", GsdPair(10.0, 10.0), (8, 8), has_mask=True)
        assert tile.mask is not None
        tile.mask[0, 0, 0] = 2
        tiles.append(tile)
    source = MemorySource(tuple(tiles))
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
    source = MemorySource((bad, *rest))
    with pytest.raises(ValueError, match="non-finite"):
        build_dataset(source, tmp_path / "ds", BuildSpec())


@pytest.mark.parametrize(
    "image",
    [
        np.full((3, 8, 8), 100, dtype=np.uint16),
        np.full((3, 8, 8), 0.5, dtype=np.float64),
        np.full((3, 8, 8), 0.5, dtype=np.complex128),
        np.full((3, 8, 8), 0.5, dtype=object),
        np.full((3, 8, 8), np.inf, dtype=np.float32),
        np.full((3, 8, 8), 1.5, dtype=np.float32),
        np.full((3, 8, 8), -0.25, dtype=np.float32),
        np.full((4, 8, 8), 0.5, dtype=np.float32),
        np.full((3, 8, 9), 0.5, dtype=np.float32),
    ],
    ids=[
        "uint16",
        "float64",
        "complex",
        "object",
        "inf",
        "above-one",
        "below-zero",
        "channel-mismatch",
        "shape-mismatch",
    ],
)
def test_prepared_image_contract_is_enforced(tmp_path: Path, image: np.ndarray) -> None:
    """Non-unit, non-float32, or mis-shaped images are rejected, not fixed."""
    gsd = GsdPair(10.0, 10.0)
    bad = replace(_tile("t0", "g0", gsd, (8, 8)), image=image)
    rest = tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8)) for index in range(1, 3))
    source = MemorySource((bad, *rest))
    with pytest.raises(ValueError):
        build_dataset(source, tmp_path / "ds", BuildSpec())


def test_non_unit_domain_is_rejected(tmp_path: Path) -> None:
    """A source that still declares DN pixels cannot build."""
    tiles = tuple(
        _tile(f"t{index}", f"g{index}", GsdPair(10.0, 10.0), (8, 8)) for index in range(3)
    )
    with pytest.raises(ValueError, match="domain"):
        build_dataset(MemorySource(tiles, domain="dn"), tmp_path / "ds", BuildSpec())


def test_stream_ref_must_match_index(tmp_path: Path) -> None:
    """A streamed row whose ref differs from its index row is rejected."""
    source = MutatedRefSource(_paired_bins())
    with pytest.raises(ValueError, match="order mismatch"):
        build_dataset(source, tmp_path / "ds", BuildSpec())


def test_anisotropic_square_tile_keeps_axis_preserving_elements(tmp_path: Path) -> None:
    """Anisotropic stored GSD drops axis-swapping elements even on square tiles."""
    gsd = GsdPair(10.0, 20.0)
    tiles = tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8), has_mask=True) for index in range(3))
    dest = tmp_path / "ds"
    build_dataset(MemorySource(tiles), dest, BuildSpec())
    rows = _rows_by_split(dest, "classifier")
    assert {row.element for row in rows["train"]} == set(SHAPE_PRESERVING)


def test_existing_destination_is_unchanged(tmp_path: Path) -> None:
    """A second build refuses to replace a finished dataset."""
    tiles = _paired_bins()
    dest = tmp_path / "ds"
    first = build_dataset(MemorySource(tiles), dest, BuildSpec())
    with pytest.raises(FileExistsError):
        build_dataset(MemorySource(tiles), dest, BuildSpec())
    assert load_manifest(dest / "dataset.json").dataset_hash == first.dataset_hash


def test_band_mismatch_is_rejected(tmp_path: Path) -> None:
    """Source bands must equal an explicit spec band list."""
    tiles = tuple(
        _tile(f"t{index}", f"g{index}", GsdPair(10.0, 10.0), (8, 8)) for index in range(3)
    )
    source = MemorySource(tiles, band_names=("NIR", "RED", "GREEN"))
    with pytest.raises(ValueError, match="band_names"):
        build_dataset(source, tmp_path / "ds", BuildSpec(input_bands=("BLUE", "GREEN", "RED")))


def test_manifest_bands_resolve_from_source(tmp_path: Path) -> None:
    """A spec without input_bands records the source band list."""
    tiles = tuple(
        _tile(f"t{index}", f"g{index}", GsdPair(10.0, 10.0), (8, 8), channels=2)
        for index in range(3)
    )
    dest = tmp_path / "ds"
    manifest = build_dataset(MemorySource(tiles, band_names=("A", "B")), dest, BuildSpec())
    assert manifest.band_names == ("A", "B")
    manifest = load_manifest(dest / "dataset.json")
    assert manifest.band_names == ("A", "B")
    shard = next((dest / "classifier" / "train").iterdir())
    dataset = ShardDataset(shard, manifest.gsd_reference_m, "classifier", channels=2)
    image, _gsd, _target = dataset[0]
    assert image.shape == (2, 8, 8)


def test_nonpositive_gsd_is_rejected(tmp_path: Path) -> None:
    """A zero GSD component is rejected."""
    tiles = tuple(_tile(f"t{index}", f"g{index}", GsdPair(0.0, 10.0), (8, 8)) for index in range(3))
    with pytest.raises(ValueError, match="gsd"):
        build_dataset(MemorySource(tiles), tmp_path / "ds", BuildSpec())


def test_stream_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    """Pixel size must match the indexed dimensions."""
    gsd = GsdPair(10.0, 10.0)
    bad = replace(
        _tile("t0", "g0", gsd, (8, 8)),
        ref=replace(_tile("t0", "g0", gsd, (8, 8)).ref, height=10, width=10),
    )
    rest = tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8)) for index in range(1, 3))
    source = MemorySource((bad, *rest))
    with pytest.raises(ValueError, match="indexed size"):
        build_dataset(source, tmp_path / "ds", BuildSpec())


def test_unit_pixels_preserved_exactly(tmp_path: Path) -> None:
    """A non-grid-aligned float32 survives build -> npy -> loader exactly."""
    gsd = GsdPair(10.0, 10.0)
    marker = _tile("t0", "g0", gsd, (8, 8))
    marker.image[0, 3, 4] = np.float32(0.1234567)
    marker.image[1, 0, 0] = np.float32(0.0)
    marker.image[2, 7, 7] = np.float32(1.0)
    tiles = (marker,) + tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8)) for index in range(1, 3))
    source_image = marker.image.copy()
    dest = tmp_path / "ds"
    build_dataset(MemorySource(tiles), dest, BuildSpec())
    np.testing.assert_array_equal(marker.image, source_image)
    found = False
    for shard in _shard_dirs(dest, "classifier"):
        rows = read_rows(shard)
        index = next(
            (i for i, row in enumerate(rows) if row.tile_id == "t0" and row.element == "id"),
            None,
        )
        if index is None:
            continue
        found = True
        stored = read_images(shard)[index]
        np.testing.assert_array_equal(stored, source_image)
        assert stored.dtype == np.float32
        dataset = ShardDataset(shard, 15.87, "classifier", channels=3)
        image, _g, _t = dataset[index]
        np.testing.assert_array_equal(image.numpy(), source_image)
        assert float(image[0, 3, 4]) == float(np.float32(0.1234567))
    assert found


def test_augmented_rows_only_permute_pixels(tmp_path: Path) -> None:
    """Flips and rotations permute stored pixels and aligned masks."""
    gsd = GsdPair(10.0, 10.0)
    image = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8) / np.float32(255.0)
    mask = np.zeros((1, 8, 8), dtype=np.uint8)
    mask[0, :4, :] = 1
    tiles = tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8), has_mask=True) for index in range(3))
    index = assign_group_splits([tile.ref.group_id for tile in tiles], SplitRecipe())
    train_row = index.train[0]
    marker = replace(tiles[train_row], image=image, mask=mask)
    tiles = tuple(marker if position == train_row else tile for position, tile in enumerate(tiles))
    dest = tmp_path / "ds"
    build_dataset(MemorySource(tiles), dest, BuildSpec())
    seen = 0
    for shard in _shard_dirs(dest, "segmentor"):
        rows = read_rows(shard)
        images = read_images(shard)
        masks = read_masks(shard)
        assert masks is not None
        for row_index, row in enumerate(rows):
            if row.tile_id != marker.ref.tile_id:
                continue
            seen += 1
            np.testing.assert_array_equal(images[row_index], apply_dihedral(image, row.element))
            np.testing.assert_array_equal(
                masks[row_index], apply_dihedral(mask, row.element).astype(np.uint8)
            )
    assert seen == len(ELEMENT_NAMES)


def test_indexed_shape_is_not_derived_from_gsd(tmp_path: Path) -> None:
    """Stored GSD is the row's measured value, not extent over pixels."""
    gsd = GsdPair(12.34, 56.78)
    tiles = tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8)) for index in range(3))
    dest = tmp_path / "ds"
    build_dataset(MemorySource(tiles), dest, BuildSpec())
    for shard in _shard_dirs(dest, "classifier"):
        assert shard.name == "8x8"
        stored = read_gsd(shard)
        np.testing.assert_allclose(stored[0], [12.34, 56.78], rtol=1e-6)


def test_schema1_dataset_is_rejected(tmp_path: Path) -> None:
    """A schema-1 manifest fails with a rebuild message, before image reads."""
    dest = tmp_path / "ds"
    build_dataset(MemorySource(_paired_bins()), dest, BuildSpec())
    path = dest / "dataset.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = 1
    payload["image_scale"] = 65535
    del payload["image_dtype"]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema 1.*rebuild|rebuild.*schema"):
        load_manifest(path)


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
        "image": np.full((3, 193, 258), 0.5, dtype=np.float32),
    }
    fields.update(overrides)
    return FlightTileWrite(**fields)  # type: ignore[arg-type]


def test_build_flight_rejects_nominal_rows(tmp_path: Path) -> None:
    """The standard flight build refuses nominal-GSD captures."""
    source_dir = tmp_path / "flight-src"
    write_flight_tile_dir(source_dir, [_flight_tile(0), _flight_tile(1, gsd_nominal=True)])
    with pytest.raises(ValueError, match="nominal"):
        build_flight(source_dir, tmp_path / "ds")


def test_build_flight_round_trip_preserves_pixels_and_metadata(tmp_path: Path) -> None:
    """Flight import pixels, GSD, groups, and nominal tags reach the dataset."""
    image = np.full((3, 193, 258), np.float32(0.1234567), dtype=np.float32)
    source_dir = tmp_path / "flight-src"
    write_flight_tile_dir(
        source_dir,
        [
            _flight_tile(0, image=image.copy(), frame_id="f0", group_id="g0"),
            _flight_tile(1, frame_id="f1", group_id="g1"),
            _flight_tile(2, frame_id="f2", group_id="g2"),
            _flight_tile(3, frame_id="f3", group_id="g3"),
        ],
        source_ref="flight-test",
    )
    dest = tmp_path / "ds"
    manifest = build_flight(source_dir, dest)
    assert manifest.source == "flight"
    assert manifest.source_ref == "flight-test"
    assert manifest.image_dtype == "float32"
    found = False
    for shard in _shard_dirs(dest, "classifier"):
        rows = read_rows(shard)
        index = next(
            (i for i, row in enumerate(rows) if row.tile_id == "f0-0-0" and row.element == "id"),
            None,
        )
        if index is None:
            continue
        found = True
        np.testing.assert_array_equal(read_images(shard)[index], image)
        assert rows[index].bin_id == "elevation15"
    assert found


def _research_tile(tile_id: str, group_id: str, **ref_overrides: object) -> RawTile:
    """One 4x8 research-source tile with overridable ref fields."""
    fields: dict[str, object] = {
        "tile_id": tile_id,
        "group_id": group_id,
        "label": 1.0,
        "has_mask": False,
        "gsd": GsdPair(10.0, 20.0),
        "height": 4,
        "width": 8,
        "frame_id": group_id,
        "grid_rc": (0, 0),
        "bin_id": "",
    }
    fields.update(ref_overrides)
    return RawTile(
        ref=RawTileRef(**fields),  # type: ignore[arg-type]
        image=np.full((3, 4, 8), 0.25, dtype=np.float32),
        mask=None,
    )


def test_research_source_records_nominal_flag(tmp_path: Path) -> None:
    """A custom research source keeps gsd_nominal and theta on the row."""
    tiles = tuple(_research_tile(f"t{index}", f"g{index}") for index in range(3)) + (
        _research_tile("t-nominal", "g0", theta_g_deg=30.0, gsd_nominal=True),
    )
    source = MemorySource(tiles)
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
    source = MemorySource(tiles)
    with pytest.raises(ValueError, match="theta_g_deg"):
        build_dataset(source, tmp_path / "ds", BuildSpec())


def _meta_tile(
    tile_id: str,
    group_id: str,
    bin_id: str,
    metadata: ObservationMetadata,
    mask: np.ndarray | None,
    *,
    label: float = 1.0,
    has_mask: bool | None = None,
    gsd: GsdPair = GsdPair(10.0, 10.0),
) -> RawTile:
    """One 4x4 isotropic-GSD tile carrying explicit observation metadata."""
    return RawTile(
        ref=RawTileRef(
            tile_id=tile_id,
            group_id=group_id,
            label=label,
            has_mask=mask is not None if has_mask is None else has_mask,
            gsd=gsd,
            height=4,
            width=4,
            frame_id=None,
            grid_rc=None,
            bin_id=bin_id,
            metadata=metadata,
        ),
        image=np.full((3, 4, 4), 0.5, dtype=np.float32),
        mask=mask,
    )


def _observed_groups() -> tuple[tuple[RawTile, ...], dict[str, tuple[ObservationMetadata, str]]]:
    """Three groups, two GSD variants each, one observation id per group."""
    positive = np.zeros((1, 4, 4), dtype=np.uint8)
    positive[0, 0, 0] = 1
    empty = np.zeros((1, 4, 4), dtype=np.uint8)
    expected: dict[str, tuple[ObservationMetadata, str]] = {
        "g0": (
            ObservationMetadata(
                observation_id="obs-g0",
                acquired_at_utc="2020-02-29T10:56:41Z",
                conditions=(ConditionTag(name="weather", value="recorded-clear"),),
                annotation_source="labels/obs-g0.json",
                annotation_version="r1",
                source_annotation_state="NONEMPTY",
            ),
            "NONEMPTY",
        ),
        "g1": (
            ObservationMetadata(
                observation_id="obs-g1",
                annotation_source="labels/obs-g1.json",
                source_annotation_state="EXPLICIT_EMPTY",
            ),
            "EMPTY",
        ),
        "g2": (
            ObservationMetadata(
                observation_id="obs-g2",
                annotation_source="labels/obs-g2.json",
                source_annotation_state="NONEMPTY",
            ),
            "EMPTY",
        ),
    }
    masks = {"g0": positive, "g1": empty, "g2": empty}
    tiles = tuple(
        _meta_tile(
            f"{group}-{bin_id}",
            group,
            bin_id,
            expected[group][0],
            masks[group],
            gsd=gsd,
        )
        for group in ("g0", "g1", "g2")
        for bin_id, gsd in (("low", GsdPair(10.0, 10.0)), ("high", GsdPair(20.0, 20.0)))
    )
    return tiles, expected


def test_observation_metadata_shared_across_variants_tasks_and_elements(
    tmp_path: Path,
) -> None:
    """One observation id survives both bins, both tasks, and every element."""
    tiles, expected = _observed_groups()
    dest = tmp_path / "ds"
    build_dataset(MemorySource(tiles), dest, BuildSpec())
    seen_ids: set[str] = set()
    group_states: dict[str, set[str]] = {}
    for task in ("classifier", "segmentor"):
        for shard in _shard_dirs(dest, task):
            rows = read_rows(shard)
            stored_gsd = read_gsd(shard)
            for row_index, row in enumerate(rows):
                metadata, _state = expected[row.group_id]
                assert row.metadata == metadata
                expected_pair = (10.0, 10.0) if row.bin_id == "low" else (20.0, 20.0)
                np.testing.assert_allclose(stored_gsd[row_index], expected_pair, rtol=1e-6)
                seen_ids.add(cast(str, row.metadata.observation_id))
                group_states.setdefault(row.group_id, set()).add(row.prepared_mask_state)
    assert seen_ids == {"obs-g0", "obs-g1", "obs-g2"}
    assert group_states["g0"] == {"NONEMPTY"}
    assert group_states["g1"] == {"EMPTY"}
    assert group_states["g2"] == {"EMPTY"}
    train_rows = _rows_by_split(dest, "classifier")["train"]
    assert {row.element for row in train_rows} == set(ELEMENT_NAMES)
    for group, (metadata, state) in expected.items():
        segmentor_rows = [
            row
            for rows in _rows_by_split(dest, "segmentor").values()
            for row in rows
            if row.group_id == group
        ]
        assert segmentor_rows, group
        assert all(row.metadata == metadata for row in segmentor_rows)
        assert all(row.prepared_mask_state == state for row in segmentor_rows)


def test_classifier_negative_missing_mask_records_prepared_missing(tmp_path: Path) -> None:
    """A classifier-only tile keeps prepared MISSING and no segmentor rows."""
    positive = np.zeros((1, 4, 4), dtype=np.uint8)
    positive[0, 0, 0] = 1
    tiles = tuple(
        _meta_tile(
            f"t{index}",
            f"g{index}",
            "",
            ObservationMetadata(observation_id=f"obs-{index}", source_annotation_state="NONEMPTY"),
            positive,
        )
        for index in range(3)
    ) + (
        _meta_tile(
            "t-neg",
            "g0",
            "",
            ObservationMetadata(observation_id="obs-neg", source_annotation_state="MISSING"),
            None,
            label=0.0,
        ),
    )
    dest = tmp_path / "ds"
    build_dataset(MemorySource(tiles), dest, BuildSpec())
    rows = [
        row
        for rows in _rows_by_split(dest, "classifier").values()
        for row in rows
        if row.tile_id == "t-neg"
    ]
    assert rows
    assert all(row.prepared_mask_state == "MISSING" for row in rows)
    assert all(row.metadata.observation_id == "obs-neg" for row in rows)
    assert not [
        row
        for rows in _rows_by_split(dest, "segmentor").values()
        for row in rows
        if row.tile_id == "t-neg"
    ]


def test_explicit_empty_source_without_mask_records_prepared_missing(tmp_path: Path) -> None:
    """A known explicit-empty annotation with no prepared mask stays legal."""
    positive = np.zeros((1, 4, 4), dtype=np.uint8)
    positive[0, 0, 0] = 1
    tiles = tuple(
        _meta_tile(
            f"t{index}",
            f"g{index}",
            "",
            ObservationMetadata(observation_id=f"obs-{index}", source_annotation_state="NONEMPTY"),
            positive,
        )
        for index in range(3)
    ) + (
        _meta_tile(
            "t-empty",
            "g0",
            "",
            ObservationMetadata(
                observation_id="obs-empty", source_annotation_state="EXPLICIT_EMPTY"
            ),
            None,
            label=0.0,
        ),
    )
    dest = tmp_path / "ds"
    build_dataset(MemorySource(tiles), dest, BuildSpec())
    rows = [
        row
        for rows in _rows_by_split(dest, "classifier").values()
        for row in rows
        if row.tile_id == "t-empty"
    ]
    assert rows
    assert all(row.prepared_mask_state == "MISSING" for row in rows)
    assert not [
        row
        for rows in _rows_by_split(dest, "segmentor").values()
        for row in rows
        if row.tile_id == "t-empty"
    ]


def test_source_annotation_mask_contradictions_are_rejected(tmp_path: Path) -> None:
    """A mask contradicting its recorded source annotation state fails the build."""
    positive = np.zeros((1, 4, 4), dtype=np.uint8)
    positive[0, 0, 0] = 1
    good = tuple(
        _meta_tile(
            f"t{index}",
            f"g{index}",
            "",
            ObservationMetadata(observation_id=f"obs-{index}"),
            positive,
        )
        for index in range(2)
    )
    missing_with_mask = _meta_tile(
        "t-bad",
        "g2",
        "",
        ObservationMetadata(observation_id="obs-bad", source_annotation_state="MISSING"),
        positive,
    )
    with pytest.raises(ValueError, match="MISSING"):
        build_dataset(MemorySource(good + (missing_with_mask,)), tmp_path / "a", BuildSpec())
    empty_with_positive_mask = _meta_tile(
        "t-bad",
        "g2",
        "",
        ObservationMetadata(observation_id="obs-bad", source_annotation_state="EXPLICIT_EMPTY"),
        positive,
    )
    with pytest.raises(ValueError, match="EXPLICIT_EMPTY"):
        build_dataset(MemorySource(good + (empty_with_positive_mask,)), tmp_path / "b", BuildSpec())


def test_schema2_fixture_reads_unchanged_with_unknown_metadata(tmp_path: Path) -> None:
    """A schema-2 fixture reads with unknown metadata and is never mutated."""
    tiles = _paired_bins()
    dest = tmp_path / "ds"
    build_dataset(MemorySource(tiles), dest, BuildSpec())
    for rows_path in sorted(dest.rglob("rows.jsonl")):
        lines = []
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            payload.pop("metadata", None)
            payload.pop("prepared_mask_state", None)
            lines.append(json.dumps(payload, separators=(",", ":")))
        rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path = dest / "dataset.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["schema"] = 2
    manifest_payload["dataset_hash"] = compute_dataset_hash(dest)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    before = {
        path.relative_to(dest).as_posix(): path.read_bytes()
        for path in dest.rglob("*")
        if path.is_file()
    }
    loaded = load_manifest(manifest_path)
    assert loaded.schema_version == 2
    for task in ("classifier", "segmentor"):
        for split_name in ("train", "val", "test"):
            split_dir = dest / task / split_name
            if not split_dir.is_dir():
                continue
            for shard in split_dir.iterdir():
                for row in read_rows(shard):
                    assert row.metadata == ObservationMetadata()
                    assert row.prepared_mask_state == "UNKNOWN"
    after = {
        path.relative_to(dest).as_posix(): path.read_bytes()
        for path in dest.rglob("*")
        if path.is_file()
    }
    assert before == after
    fresh = tmp_path / "ds3"
    build_dataset(MemorySource(tiles), fresh, BuildSpec())
    assert load_manifest(fresh / "dataset.json").schema_version == 3
    for task in ("classifier", "segmentor"):
        for split_name in ("train", "val", "test"):
            old_dir = dest / task / split_name
            if not old_dir.is_dir():
                continue
            for shard in sorted(old_dir.iterdir()):
                twin = fresh / task / split_name / shard.name
                np.testing.assert_array_equal(read_images(shard), read_images(twin))
                np.testing.assert_array_equal(read_gsd(shard), read_gsd(twin))
                np.testing.assert_array_equal(read_labels(shard), read_labels(twin))
                old_masks, new_masks = read_masks(shard), read_masks(twin)
                assert (old_masks is None) == (new_masks is None)
                if old_masks is not None and new_masks is not None:
                    np.testing.assert_array_equal(old_masks, new_masks)
                dataset = ShardDataset(shard, loaded.gsd_reference_m, task, channels=3)
                image, _gsd, target = dataset[0]
                assert image.shape[0] == 3
                assert target.shape[0] == 1
    manifest_payload["schema"] = 4
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_manifest(manifest_path)
