"""Tests for shard loading."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from tools.ml_models.dataset.build import build_dataset
from tools.ml_models.dataset.loader import ShardDataset, make_loader
from tools.ml_models.dataset.preprocess import dequantize_unit
from tools.ml_models.dataset.raw import BinSpec, GsdPair, RawTile, RawTileRef
from tools.ml_models.dataset.spec import BuildSpec
from tools.ml_models.dataset.store import read_images
from torch.utils.data import DataLoader, Subset


class _Source:
    """Small raw source used by loader tests."""

    def __init__(self, tiles: tuple[RawTile, ...]) -> None:
        self.name = "memory"
        self.band_names: tuple[str, ...] = ("BLUE", "GREEN", "RED")
        self.domain = "dn"
        self.bit_depth = 12
        self.source_ref = "test"
        self.extent_m: tuple[float, float] | None = (80.0, 80.0)
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
    """Build one DN tile."""
    height, width = shape
    image = np.full((3, height, width), 200, dtype=np.uint16)
    return RawTile(
        ref=RawTileRef(
            tile_id=tile_id,
            group_id=group_id,
            label=label,
            has_mask=False,
            gsd=gsd,
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


def test_uint16_reads_back_as_unit(tmp_path: Path) -> None:
    """A shard row dequantizes to the value ShardDataset returns."""
    dest = _build(tmp_path, "ds", _square(1.0))
    shard = next((dest / "classifier" / "train").iterdir())
    dataset = ShardDataset(shard, 15.87, "classifier")
    image, gsd, target = dataset[0]
    stored = dequantize_unit(read_images(shard)[0])
    np.testing.assert_allclose(image.numpy(), stored)
    assert gsd.shape == (2,)
    assert gsd.dtype == torch.float32
    assert target.shape == (1,)


def test_shard_dataset_batches_through_dataloader(tmp_path: Path) -> None:
    """ShardDataset stacks through a torch DataLoader over a Subset."""
    dest = _build(tmp_path, "ds", _square(1.0))
    shard = next((dest / "classifier" / "train").iterdir())
    dataset = ShardDataset(shard, 15.87, "classifier")
    loader = DataLoader(Subset(dataset, [0, 1]), batch_size=2, shuffle=False)
    image, gsd, target = next(iter(loader))
    assert image.shape == (2, 3, 8, 8)
    assert gsd.shape == (2, 2)
    assert target.shape == (2, 1)


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
    for image, _gsd, _target in make_loader(
        [dest], "classifier", "train", 2, None, 0, n_batches=24
    ):
        assert image.shape[0] == 2
        shapes.add((int(image.shape[-2]), int(image.shape[-1])))
        assert len({(int(image.shape[-2]), int(image.shape[-1]))}) == 1
    assert shapes == {(8, 8), (4, 8)}


def test_weights_and_seed_fix_the_order(tmp_path: Path) -> None:
    """Equal seeds match. A heavier dataset contributes more batches."""
    low = _build(tmp_path, "low", _square(0.0))
    high = _build(tmp_path, "high", _square(1.0))
    first = list(make_loader([low, high], "classifier", "train", 4, (1.0, 9.0), 3, n_batches=40))
    second = list(make_loader([low, high], "classifier", "train", 4, (1.0, 9.0), 3, n_batches=40))
    third = list(make_loader([low, high], "classifier", "train", 4, (1.0, 9.0), 4, n_batches=40))
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left[0], right[0])
        torch.testing.assert_close(left[2], right[2])
    positives = sum(int(batch[2].mean().item() > 0.5) for batch in first)
    assert positives > 28
    assert not torch.equal(first[0][2], third[0][2]) or not torch.equal(first[1][2], third[1][2])
