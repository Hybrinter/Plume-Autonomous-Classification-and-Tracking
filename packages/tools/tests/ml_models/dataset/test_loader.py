"""Tests for shard loading."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from tools.ml_models.dataset.build import build_dataset
from tools.ml_models.dataset.loader import ShardDataset, make_loader
from tools.ml_models.dataset.raw import BinSpec, GsdPair, RawTile, RawTileRef
from tools.ml_models.dataset.spec import BuildSpec
from tools.ml_models.dataset.store import read_images
from torch.utils.data import DataLoader, Subset


class _Source:
    """Small raw source used by loader tests."""

    def __init__(self, tiles: tuple[RawTile, ...]) -> None:
        self.name = "memory"
        self.band_names: tuple[str, ...] = ("BLUE", "GREEN", "RED")
        self.domain = "unit"
        self.source_ref = "test"
        self.bins: tuple[BinSpec, ...] = ()
        self._tiles = tiles

    def index(self) -> tuple[RawTileRef, ...]:
        """Return refs in stream order."""
        return tuple(tile.ref for tile in self._tiles)

    def iter_tiles(self) -> Iterator[RawTile]:
        """Yield the in-memory tiles."""
        yield from self._tiles


def _tile(
    tile_id: str, group_id: str, gsd: GsdPair, shape: tuple[int, int], label: float
) -> RawTile:
    """Build one unit tile."""
    height, width = shape
    image = np.full((3, height, width), 0.25, dtype=np.float32)
    return RawTile(
        ref=RawTileRef(
            tile_id=tile_id,
            group_id=group_id,
            label=label,
            has_mask=False,
            gsd=gsd,
            height=height,
            width=width,
            frame_id=group_id,
            grid_rc=None,
            bin_id="",
        ),
        image=image,
        mask=None,
    )


def _square(label: float) -> tuple[RawTile, ...]:
    """Three square tiles with one label."""
    gsd = GsdPair(10.0, 10.0)
    return tuple(_tile(f"t{index}", f"g{index}", gsd, (8, 8), label) for index in range(3))


def _build(tmp_path: Path, name: str, tiles: tuple[RawTile, ...]) -> Path:
    """Build one small dataset and return its directory."""
    dest = tmp_path / name
    build_dataset(_Source(tiles), dest, BuildSpec())
    return dest


def test_float32_reads_back_exactly(tmp_path: Path) -> None:
    """A shard row returns the stored float32 pixels unchanged."""
    dest = _build(tmp_path, "ds", _square(1.0))
    shard = next((dest / "classifier" / "train").iterdir())
    dataset = ShardDataset(shard, 15.87, "classifier", channels=3)
    image, gsd, target = dataset[0]
    stored = read_images(shard)[0]
    np.testing.assert_array_equal(image.numpy(), stored)
    assert image.dtype == torch.float32
    assert gsd.shape == (2,)
    assert gsd.dtype == torch.float32
    assert target.shape == (1,)


def test_shard_dataset_batches_through_dataloader(tmp_path: Path) -> None:
    """ShardDataset stacks through a torch DataLoader over a Subset."""
    dest = _build(tmp_path, "ds", _square(1.0))
    shard = next((dest / "classifier" / "train").iterdir())
    dataset = ShardDataset(shard, 15.87, "classifier", channels=3)
    loader = DataLoader(Subset(dataset, [0, 1]), batch_size=2, shuffle=False)
    image, gsd, target = next(iter(loader))
    assert image.shape == (2, 3, 8, 8)
    assert gsd.shape == (2, 2)
    assert target.shape == (2, 1)


def test_wrong_dtype_or_channel_layout_is_rejected(tmp_path: Path) -> None:
    """images.npy must be float32 with the declared channel count."""
    dest = _build(tmp_path, "ds", _square(1.0))
    shard = next((dest / "classifier" / "train").iterdir())
    np.save(shard / "images.npy", np.zeros((3, 3, 8, 8), dtype=np.uint16))
    with pytest.raises(ValueError, match="float32"):
        ShardDataset(shard, 15.87, "classifier", channels=3)
    np.save(shard / "images.npy", np.zeros((3, 2, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        ShardDataset(shard, 15.87, "classifier", channels=3)


def test_fortran_storage_returns_c_contiguous_copies(tmp_path: Path) -> None:
    """A Fortran-ordered images.npy yields exact C-contiguous pixel copies."""
    dest = _build(tmp_path, "ds", _square(1.0))
    shard = next((dest / "classifier" / "train").iterdir())
    images = np.load(shard / "images.npy")
    np.save(shard / "images.npy", np.asfortranarray(images))
    dataset = ShardDataset(shard, 15.87, "classifier", channels=3)
    image, _gsd, _target = dataset[0]
    np.testing.assert_array_equal(image.numpy(), np.asarray(images[0]))
    assert image.is_contiguous()
    image[0, 0, 0] = 99.0
    np.testing.assert_array_equal(np.load(shard / "images.npy")[0], images[0])


def test_malformed_shard_layout_is_rejected(tmp_path: Path) -> None:
    """Array files must match the declared dtype, shape, and row count."""
    dest = _build(tmp_path, "ds", _square(1.0))
    shard = next((dest / "classifier" / "train").iterdir())
    originals = {name: np.load(shard / name) for name in ("images.npy", "gsd.npy", "labels.npy")}
    count, channels, height, width = originals["images.npy"].shape
    cases = [
        ("images.npy", np.zeros((0, channels, height, width), dtype=np.float32)),
        ("images.npy", np.zeros((count, channels, 0, width), dtype=np.float32)),
        ("gsd.npy", np.zeros((count, 2), dtype=np.float64)),
        ("gsd.npy", np.zeros((count, 3), dtype=np.float32)),
        ("gsd.npy", np.zeros((count + 1, 2), dtype=np.float32)),
        ("labels.npy", np.zeros((count, 1), dtype=np.float64)),
        ("labels.npy", np.zeros((count,), dtype=np.float32)),
        ("labels.npy", np.zeros((count + 1, 1), dtype=np.float32)),
        ("masks.npy", np.zeros((count, 1, height, width), dtype=np.float32)),
        ("masks.npy", np.zeros((count + 1, 1, height, width), dtype=np.uint8)),
        ("masks.npy", np.zeros((count, 2, height, width), dtype=np.uint8)),
    ]
    for name, array in cases:
        np.save(shard / name, array)
        with pytest.raises(ValueError, match=name.split(".")[0]):
            ShardDataset(shard, 15.87, "classifier", channels=channels)
        if name in originals:
            np.save(shard / name, originals[name])
        else:
            (shard / name).unlink()


def test_out_of_unit_row_is_rejected_on_read(tmp_path: Path) -> None:
    """A stored row outside [0, 1] fails when the row is loaded."""
    dest = _build(tmp_path, "ds", _square(1.0))
    shard = next((dest / "classifier" / "train").iterdir())
    images = np.load(shard / "images.npy")
    images[0, 0, 0, 0] = np.float32(1.5)
    np.save(shard / "images.npy", images)
    dataset = ShardDataset(shard, 15.87, "classifier", channels=3)
    with pytest.raises(ValueError, match="unit"):
        dataset[0]


def test_batches_stay_inside_one_shard(tmp_path: Path) -> None:
    """Each batch has one spatial size, and both sizes appear."""
    gsd_square = GsdPair(10.0, 10.0)
    gsd_rect = GsdPair(10.0, 20.0)
    tiles: list[RawTile] = []
    for index in range(3):
        tiles.append(_tile(f"s{index}", f"g{index}", gsd_square, (8, 8), 1.0))
        tiles.append(_tile(f"r{index}", f"g{index}", gsd_rect, (4, 8), 0.0))
    dest = _build(tmp_path, "mixed", tuple(tiles))
    shapes: set[tuple[int, int]] = set()
    for image, _gsd, _target in make_loader(dest, "classifier", "train", 2, 0, n_batches=24):
        assert image.shape[0] == 2
        shapes.add((int(image.shape[-2]), int(image.shape[-1])))
        assert len({(int(image.shape[-2]), int(image.shape[-1]))}) == 1
    assert shapes == {(8, 8), (4, 8)}


def test_seed_fixes_the_batch_order(tmp_path: Path) -> None:
    """Equal seeds produce identical batches; a different seed differs."""
    gsd = GsdPair(10.0, 10.0)
    tiles = _square(1.0) + tuple(
        _tile(f"u{index}", f"h{index}", gsd, (8, 8), 0.0) for index in range(3)
    )
    dest = _build(tmp_path, "ds", tiles)
    first = list(make_loader(dest, "classifier", "train", 4, 3, n_batches=40))
    second = list(make_loader(dest, "classifier", "train", 4, 3, n_batches=40))
    third = list(make_loader(dest, "classifier", "train", 4, 4, n_batches=40))
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left[0], right[0])
        torch.testing.assert_close(left[2], right[2])
    assert not torch.equal(first[0][2], third[0][2]) or not torch.equal(first[1][2], third[1][2])
