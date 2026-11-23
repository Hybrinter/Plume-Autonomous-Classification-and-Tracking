"""Tests for the two-input calibration batch collector."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.dataset.build import build_dataset
from tools.ml_models.dataset.gsd import to_model_gsd
from tools.ml_models.dataset.raw import BinSpec, GsdPair, RawTile, RawTileRef
from tools.ml_models.dataset.spec import BuildSpec
from tools.ml_models.dataset.store import RowRecord, read_gsd, read_images, read_rows
from tools.ml_models.export.calibration import calibration_batches
from tools.ml_models.export.contract import CONDITIONING_ID, GSD_ENCODING
from tools.ml_models.export.manifest import ModelManifest


class MemorySource:
    """In-memory raw source over small tiles."""

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
        """Yield each tile once."""
        yield from self._tiles


def _tile(group: str, index: int, gsd: GsdPair) -> RawTile:
    """Build one raw tile in ``group`` sized by its paired GSD."""
    height = round(80.0 / gsd.along_m)
    width = round(80.0 / gsd.lateral_m)
    fill = np.float32(0.1234567) if index == 0 else np.float32((index + 1) / 10.0)
    ref = RawTileRef(
        tile_id=f"{group}-{index}",
        group_id=group,
        label=float(index % 2),
        has_mask=False,
        gsd=gsd,
        height=height,
        width=width,
        frame_id=group,
        grid_rc=(0, 0),
        bin_id="",
    )
    return RawTile(
        ref=ref,
        image=np.full((3, height, width), fill, dtype=np.float32),
        mask=None,
    )


def _dataset(dest: Path, gsd: GsdPair = GsdPair(10.0, 20.0)) -> Path:
    """Build a small finished dataset with ten groups."""
    tiles = tuple(_tile(f"g{group}", index, gsd) for group in range(10) for index in range(2))
    build_dataset(MemorySource(tiles), dest, BuildSpec())
    return dest


def _model(kind: str = "classifier") -> ModelManifest:
    """A conforming sidecar for the calibration gate."""
    arch = "pactnet_stub" if kind == "classifier" else "dilatenet_stub"
    output = (None, 1) if kind == "classifier" else (None, 1, None, None)
    return ModelManifest(
        version="a" * 16,
        kind=kind,  # type: ignore[arg-type]
        arch=arch,
        sha256="0" * 64,
        dataset_hash="b" * 64,
        band_names=("BLUE", "GREEN", "RED"),
        input_shape=(None, 3, None, None),
        gsd_input_shape=(None, 2),
        output_shape=output,
        gsd_reference_m=15.87,
        gsd_min_m=(9.0, 19.0),
        gsd_max_m=(40.0, 60.0),
        conditioning=CONDITIONING_ID,
        gsd_encoding=GSD_ENCODING,
    )


def _train_shard_rows(dest: Path, task: str = "classifier") -> tuple[list[RowRecord], Path]:
    """Return the train rows and shard directory for ``task``."""
    task_dir = dest / task / "train"
    shards = sorted(task_dir.iterdir())
    rows: list[RowRecord] = []
    for shard in shards:
        rows.extend(read_rows(shard))
    return rows, shards[0]


def test_batches_feed_image_and_gsd_pairs(tmp_path: Path) -> None:
    """Every batch carries a float32 (1,C,H,W) image and (1,2) encoded GSD."""
    dest = _dataset(tmp_path / "ds")
    rows, shard_dir = _train_shard_rows(dest)
    batches = calibration_batches(dest, _model(), samples=32)
    assert len(batches) == min(32, len(rows))
    raw_gsd = read_gsd(shard_dir)
    stored = read_images(shard_dir)
    for index, batch in enumerate(batches):
        assert set(batch) == {"image", "gsd"}
        assert batch["image"].shape == (1, 3, 4, 8)
        assert batch["image"].dtype == np.float32
        assert batch["gsd"].shape == (1, 2)
        assert batch["gsd"].dtype == np.float32
        np.testing.assert_array_equal(batch["image"][0], stored[index])
        expected = to_model_gsd(np.asarray(raw_gsd[index]), 15.87)
        np.testing.assert_allclose(batch["gsd"][0], expected, rtol=1e-6)


def test_batches_round_robin_across_shards(tmp_path: Path) -> None:
    """Rows interleave one per same-root shard per pass, in train order."""
    dest = tmp_path / "ds"
    tiles = tuple(
        _tile(f"g{group}", index, GsdPair(10.0, 20.0) if group % 2 == 0 else GsdPair(15.0, 30.0))
        for group in range(10)
        for index in range(2)
    )
    build_dataset(MemorySource(tiles), dest, BuildSpec())
    task_dir = dest / "classifier" / "train"
    shard_dirs = sorted(task_dir.iterdir())
    assert len(shard_dirs) == 2
    shard_gsd = [read_gsd(shard_dir) for shard_dir in shard_dirs]
    batches = calibration_batches(dest, _model(), samples=6)
    assert len(batches) == 6
    expected = to_model_gsd(
        np.asarray(
            [
                shard_gsd[0][0],
                shard_gsd[1][0],
                shard_gsd[0][1],
                shard_gsd[1][1],
                shard_gsd[0][2],
                shard_gsd[1][2],
            ]
        ),
        15.87,
    )
    np.testing.assert_allclose(
        np.concatenate([batch["gsd"] for batch in batches]), expected, rtol=1e-6
    )


def test_calibration_requires_inputs_and_train_rows(tmp_path: Path) -> None:
    """A bad sample count and missing shards are rejected."""
    dest = _dataset(tmp_path / "ds")
    model = _model()
    with pytest.raises(ValueError, match="positive count"):
        calibration_batches(dest, model, samples=0)
    with pytest.raises(ValueError, match="no training shards"):
        calibration_batches(dest, _model(kind="segmentor"))


def test_calibration_rejects_mismatched_preprocessing(tmp_path: Path) -> None:
    """A dataset with different bands or GSD reference is rejected."""
    dest = _dataset(tmp_path / "ds")
    model = _model()
    object.__setattr__(model, "band_names", ("BLUE", "GREEN", "NIR"))
    with pytest.raises(ValueError, match="preprocessing"):
        calibration_batches(dest, model)
    model = _model()
    object.__setattr__(model, "gsd_reference_m", 10.0)
    with pytest.raises(ValueError, match="preprocessing"):
        calibration_batches(dest, model)


def test_calibration_rejects_bad_dataset_path(tmp_path: Path) -> None:
    """An empty string or a list is not a dataset root."""
    dest = _dataset(tmp_path / "ds")
    with pytest.raises(ValueError, match="exactly one finished dataset"):
        calibration_batches("", _model())
    with pytest.raises(ValueError, match="exactly one finished dataset"):
        calibration_batches([dest], _model())  # type: ignore[arg-type]


def test_calibration_rejects_fixed_input_shape_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixed model H/W mismatch fails before any shard row is read."""
    import tools.ml_models.export.calibration as calibration

    dest = _dataset(tmp_path / "ds")
    for fixed in ((None, 3, 5, 8), (None, 3, 4, 7), (None, 3, 9, 9)):
        model = _model()
        object.__setattr__(model, "input_shape", fixed)

        class _Unread:
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise AssertionError("shard opened before the shape check")

        monkeypatch.setattr(calibration, "ShardDataset", _Unread)
        with pytest.raises(ValueError, match="shard shape"):
            calibration_batches(dest, model)
