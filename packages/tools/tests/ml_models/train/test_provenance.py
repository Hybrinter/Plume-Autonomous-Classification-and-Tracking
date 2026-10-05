"""Tests for measured-GSD training provenance and split-leakage checks."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from tools.ml_models.dataset.augment import AugmentRecipe
from tools.ml_models.dataset.build import build_dataset
from tools.ml_models.dataset.manifest import DatasetManifest, load_manifest
from tools.ml_models.dataset.raw import BinSpec, GsdPair, RawTile, RawTileRef
from tools.ml_models.dataset.spec import BuildSpec
from tools.ml_models.dataset.split import SplitRecipe
from tools.ml_models.dataset.store import read_gsd, read_rows
from tools.ml_models.train.provenance import training_provenance


class MemorySource:
    """In-memory raw source for provenance tests."""

    def __init__(
        self,
        tiles: tuple[RawTile, ...],
        *,
        band_names: tuple[str, ...] = ("BLUE", "GREEN", "RED"),
        source_ref: str = "test",
        bins: tuple[BinSpec, ...] = (),
    ) -> None:
        self.name = "memory"
        self.band_names = band_names
        self.domain = "unit"
        self.source_ref = source_ref
        self.bins = bins
        self._tiles = tiles

    def index(self) -> tuple[RawTileRef, ...]:
        """Return refs in stream order."""
        return tuple(tile.ref for tile in self._tiles)

    def iter_tiles(self) -> Iterator[RawTile]:
        """Yield the tiles once."""
        yield from self._tiles


def _tile(
    tile_id: str,
    group_id: str,
    gsd: GsdPair,
    shape: tuple[int, int] = (8, 16),
    *,
    label: float = 1.0,
    has_mask: bool = True,
    bin_id: str = "",
) -> RawTile:
    """Build one small raw tile."""
    height, width = shape
    image = np.arange(3 * height * width, dtype=np.float32).reshape(3, height, width) % 1024
    image = image / np.float32(1024.0)
    mask = None
    if has_mask:
        mask = np.zeros((1, height, width), dtype=np.uint8)
        mask[:, : height // 2, :] = 1
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
            grid_rc=None,
            bin_id=bin_id,
        ),
        image=image,
        mask=mask,
    )


def _groups(
    prefix: str,
    gsd: GsdPair,
    shape: tuple[int, int] = (8, 16),
    *,
    bins: tuple[str, ...] = (),
) -> tuple[RawTile, ...]:
    """Six groups of tiles at one GSD, optionally duplicated per bin."""
    tiles: list[RawTile] = []
    for index in range(6):
        group = f"{prefix}{index}"
        if bins:
            for bin_id in bins:
                tiles.append(
                    _tile(
                        f"{group}-{bin_id}",
                        group,
                        gsd,
                        shape,
                        label=float(index < 3),
                        bin_id=bin_id,
                    )
                )
        else:
            tiles.append(_tile(group, group, gsd, shape, label=float(index < 3)))
    return tuple(tiles)


_ID_ONLY = AugmentRecipe(elements=("id",))


def _spec(seed: int = 0) -> BuildSpec:
    """Small deterministic build: identity-only train augmentation."""
    return BuildSpec(split=SplitRecipe(seed=seed), augment=_ID_ONLY)


def _build(
    dest: Path,
    tiles: tuple[RawTile, ...],
    spec: BuildSpec,
    *,
    band_names: tuple[str, ...] = ("BLUE", "GREEN", "RED"),
    source_ref: str = "test",
    bins: tuple[BinSpec, ...] = (),
) -> Path:
    """Build a finished dataset and return its directory."""
    build_dataset(
        MemorySource(
            tiles,
            band_names=band_names,
            source_ref=source_ref,
            bins=bins,
        ),
        dest,
        spec,
    )
    return dest


def _rewrite_rows(shard_dir: Path, mutate: Callable[[int, dict[str, object]], None]) -> None:
    """Apply ``mutate(index, record)`` to each rows.jsonl record in place."""
    path = shard_dir / "rows.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    for index, record in enumerate(records):
        mutate(index, record)
    path.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n"
    )


def _train_shard(dataset: Path, manifest: DatasetManifest, task: str) -> Path:
    """The shard dir of the first manifest-listed train shard for ``task``."""
    shard = next(
        entry for entry in manifest.shards if entry.task == task and entry.split == "train"
    )
    return dataset / task / "train" / f"{shard.height}x{shard.width}"


def test_group_leakage_across_splits_rejected(tmp_path: Path) -> None:
    """A group reused across train and val shards of one dataset is refused."""
    dataset = _build(tmp_path / "ds", _groups("g", GsdPair(5.0, 10.0)), _spec())
    manifest = load_manifest(dataset / "dataset.json")
    train_dir = _train_shard(dataset, manifest, "classifier")
    train_groups = {row.group_id for row in read_rows(train_dir)}
    val_shard = next(
        entry for entry in manifest.shards if entry.task == "classifier" and entry.split == "val"
    )
    val_dir = dataset / "classifier" / "val" / f"{val_shard.height}x{val_shard.width}"
    _rewrite_rows(
        val_dir,
        lambda index, record: (
            record.update(group_id=sorted(train_groups)[0]) if index == 0 else None
        ),
    )
    with pytest.raises(ValueError, match="leak"):
        training_provenance(dataset, manifest, "classifier")


def test_group_leakage_across_tasks_rejected(tmp_path: Path) -> None:
    """A group in classifier train but segmentor val is refused."""
    dataset = _build(tmp_path / "ds", _groups("g", GsdPair(5.0, 10.0)), _spec())
    manifest = load_manifest(dataset / "dataset.json")
    train_dir = _train_shard(dataset, manifest, "classifier")
    train_groups = {row.group_id for row in read_rows(train_dir)}
    val_shard = next(
        entry for entry in manifest.shards if entry.task == "segmentor" and entry.split == "val"
    )
    val_dir = dataset / "segmentor" / "val" / f"{val_shard.height}x{val_shard.width}"
    _rewrite_rows(
        val_dir,
        lambda index, record: (
            record.update(group_id=sorted(train_groups)[0]) if index == 0 else None
        ),
    )
    with pytest.raises(ValueError, match="leak"):
        training_provenance(dataset, manifest, "classifier")


def test_missing_task_rejected(tmp_path: Path) -> None:
    """A dataset with no segmentor rows fails provenance for a segmentor."""
    classifier_only = _build(
        tmp_path / "c",
        _groups("c", GsdPair(5.0, 10.0)),
        BuildSpec(split=SplitRecipe(seed=0), augment=_ID_ONLY, tasks=("classifier",)),
    )
    manifest = load_manifest(classifier_only / "dataset.json")
    with pytest.raises(ValueError, match="no samples"):
        training_provenance(classifier_only, manifest, "segmentor")


def test_nominal_rows_do_not_extend_measured_coverage(tmp_path: Path) -> None:
    """A nominal train row still counts but is absent from measured GSD bounds."""
    dataset = _build(tmp_path / "ds", _groups("g", GsdPair(5.0, 10.0)), _spec())
    manifest = load_manifest(dataset / "dataset.json")
    shard_dir = _train_shard(dataset, manifest, "classifier")
    _rewrite_rows(
        shard_dir,
        lambda index, record: record.update(gsd_nominal=True) if index == 0 else None,
    )
    gsd = read_gsd(shard_dir)
    gsd[0] = (1000.0, 1000.0)
    np.save(shard_dir / "gsd.npy", gsd)
    provenance = training_provenance(dataset, manifest, "classifier")
    count = cast(int, provenance["train_samples"])
    assert count == len(read_rows(shard_dir))
    assert provenance["gsd_min_m"] == pytest.approx([5.0, 10.0])
    assert provenance["gsd_max_m"] == pytest.approx([5.0, 10.0])


def test_all_nominal_train_rows_rejected(tmp_path: Path) -> None:
    """A train split with only nominal-GSD rows has no measured coverage."""
    dataset = _build(tmp_path / "ds", _groups("g", GsdPair(5.0, 10.0)), _spec())
    manifest = load_manifest(dataset / "dataset.json")
    shard_dir = _train_shard(dataset, manifest, "classifier")
    _rewrite_rows(shard_dir, lambda index, record: record.update(gsd_nominal=True))
    with pytest.raises(ValueError, match="measured"):
        training_provenance(dataset, manifest, "classifier")
