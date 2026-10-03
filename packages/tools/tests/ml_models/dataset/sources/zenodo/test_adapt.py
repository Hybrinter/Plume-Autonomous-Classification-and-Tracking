"""Tests for ZenodoSource and the zenodo build wrapper."""

import io
import tarfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.cli import main
from tools.ml_models.dataset.augment import AugmentRecipe
from tools.ml_models.dataset.build import build_zenodo
from tools.ml_models.dataset.manifest import load_manifest
from tools.ml_models.dataset.sources.zenodo.adapt import ZenodoSource
from tools.ml_models.dataset.sources.zenodo.bins import DEFAULT_BINS, bin_hw
from tools.ml_models.dataset.sources.zenodo.prism import load_weight_table
from tools.ml_models.dataset.spec import BuildSpec
from tools.ml_models.dataset.store import read_gsd, read_masks, read_rows


def _source(archives: tuple[Path, Path, Path]) -> ZenodoSource:
    images, labels, weights = archives
    return ZenodoSource(images, labels, load_weight_table(weights))


def _shard_dirs(root: Path) -> Iterable[tuple[str, str, Path]]:
    for task_dir in sorted(root.iterdir()):
        if task_dir.name == "dataset.json" or not task_dir.is_dir():
            continue
        for split_dir in sorted(task_dir.iterdir()):
            for shard in sorted(split_dir.iterdir()):
                yield task_dir.name, split_dir.name, shard


def test_index_covers_every_bin_per_image(archives: tuple[Path, Path, Path]) -> None:
    """Six bins per image; group and mask flags follow the archive."""
    source = _source(archives)
    refs = source.index()
    assert len(refs) == 3 * len(DEFAULT_BINS)
    assert {ref.bin_id for ref in refs} == {item.bin_id for item in DEFAULT_BINS}
    by_stem = {ref.tile_id.rsplit("-", 1)[0]: ref for ref in refs if ref.bin_id == "native10"}
    assert by_stem["10003_2020-01-01T00-00-00.000Z_0"].group_id == "10003"
    assert by_stem["10003_2020-01-01T00-00-00.000Z_0"].label == 1.0
    assert by_stem["10004_2020-01-02T00-00-00.000Z_0"].label == 0.0
    assert by_stem["10004_2020-01-02T00-00-00.000Z_0"].has_mask
    assert not by_stem["10005_2020-01-03T00-00-00.000Z_0"].has_mask
    assert source.name == "zenodo"
    assert source.band_names == ("BLUE", "GREEN", "RED")
    assert source.domain == "unit"
    assert source.weight_table_id == "ap3200t-test"
    for item in DEFAULT_BINS:
        height, width = bin_hw(item)
        ref = next(r for r in refs if r.bin_id == item.bin_id)
        assert ref.gsd.lateral_m == pytest.approx(1200.0 / width)
        assert ref.gsd.along_m == pytest.approx(1200.0 / height)


def test_stream_reads_each_image_once(
    archives: tuple[Path, Path, Path], stub_stacks: list[str]
) -> None:
    """The build makes one forward pass: three decodes for 18 tiles."""
    source = _source(archives)
    tiles = list(source.iter_tiles())
    assert len(tiles) == len(source.index())
    assert stub_stacks == ["stack0", "stack1", "stack2"]
    ref_iter = iter(source.index())
    for tile in tiles:
        assert tile.ref == next(ref_iter)
    assert tiles[0].image.shape == (3, 120, 120)
    assert np.all(tiles[0].image >= 0.0) and np.all(tiles[0].image <= 1.0)


def test_stream_image_orientation_and_mix(archives: tuple[Path, Path, Path]) -> None:
    """B2 maps to blue: its y ramp varies along H only, after downsampling."""
    source = _source(archives)
    tiles = list(source.iter_tiles())
    native = tiles[0].image
    # Band fixture: B2 is a y ramp. Blue is pure B2, so the blue plane varies
    # down H and is constant across W. Green is the constant band B3.
    blue = native[0].astype(np.float64)
    assert np.all(np.diff(blue[:, 0]) >= 0.0)
    assert blue[-2:].mean() - blue[:2].mean() > 0.9
    assert np.all(blue[:, 1:] == blue[:, :-1])
    np.testing.assert_allclose(native[1], 0.1, atol=1e-6)
    np.testing.assert_allclose(native[2], 0.15, atol=1e-6)
    g35 = next(t for t in tiles if t.ref.bin_id == "gsd35")
    assert g35.image.shape == (3, 34, 34)
    ramp = g35.image[0].astype(np.float64)
    assert np.all(np.diff(ramp[:, 0]) >= 0.0)
    assert ramp[-2:].mean() - ramp[:2].mean() > 0.9


def test_masks_scale_with_each_bin(archives: tuple[Path, Path, Path]) -> None:
    """The right-half polygon fills the right half of every output grid."""
    source = _source(archives)
    tiles = {(t.ref.tile_id.rsplit("-", 1)[0], t.ref.bin_id): t for t in source.iter_tiles()}
    positive_native = tiles[("10003_2020-01-01T00-00-00.000Z_0", "native10")]
    assert positive_native.mask is not None
    np.testing.assert_array_equal(positive_native.mask[0, :, :60], 0)
    np.testing.assert_array_equal(positive_native.mask[0, :, 60:], 1)
    positive_g35 = tiles[("10003_2020-01-01T00-00-00.000Z_0", "gsd35")]
    assert positive_g35.mask is not None
    assert positive_g35.mask.shape == (1, 34, 34)
    np.testing.assert_array_equal(positive_g35.mask[0, :, :17], 0)
    np.testing.assert_array_equal(positive_g35.mask[0, :, 17:], 1)
    negative_native = tiles[("10004_2020-01-02T00-00-00.000Z_0", "native10")]
    assert negative_native.mask is not None
    # The 119-row source maps row 59 back through the native pad: half of its
    # subsamples land below the 50 percent polygon edge, so it is foreground.
    np.testing.assert_array_equal(negative_native.mask[0, :59, :], 0)
    np.testing.assert_array_equal(negative_native.mask[0, 59:, :], 1)
    bare = tiles[("10005_2020-01-03T00-00-00.000Z_0", "native10")]
    assert bare.mask is None


def test_build_counts_splits_and_stored_gsd(
    archives: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Classifier keeps all 18 rows; segmentor keeps the 12 annotated."""
    images, labels, weights = archives
    dest = tmp_path / "ds"
    spec = BuildSpec(augment=AugmentRecipe(elements=("id",)))
    manifest = build_zenodo(images, labels, weights, dest, spec)
    assert manifest.weight_table_id == "ap3200t-test"
    gsd35 = next(item for item in manifest.bins if item.bin_id == "gsd35")
    assert gsd35.lateral_m == 35.0 and gsd35.along_m == 35.0
    expected_gsd = {
        "native10": np.float32(10.0),
        "gsd15": np.float32(15.0),
        "gsd20": np.float32(20.0),
        "gsd25": np.float32(25.0),
        "gsd30": np.float32(30.0),
        "gsd35": np.float32(1200.0 / 34),
    }
    counts: dict[str, int] = {"classifier": 0, "segmentor": 0}
    group_splits: dict[str, set[str]] = {}
    shapes: dict[tuple[str, str], set[tuple[int, ...]]] = {}
    for task, split, shard in _shard_dirs(dest):
        rows = read_rows(shard)
        gsd = read_gsd(shard)
        counts[task] += len(rows)
        if task == "segmentor":
            assert read_masks(shard) is not None
        for row in rows:
            group_splits.setdefault(row.group_id, set()).add(split)
        for record, pair in zip(rows, gsd, strict=True):
            assert pair[0] == expected_gsd[record.bin_id]
            assert pair[1] == expected_gsd[record.bin_id]
            width = int(round(1200.0 / float(pair[0])))
            height = int(round(1200.0 / float(pair[1])))
            shapes.setdefault((record.bin_id, task), set()).add((height, width))
    assert counts == {"classifier": 18, "segmentor": 12}
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert shapes[("native10", "classifier")] == {(120, 120)}
    assert shapes[("gsd35", "classifier")] == {(34, 34)}


def test_dates_and_variants_of_one_location_share_a_split(
    archives: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """A second unannotated 10003 date stays in the location's single split."""
    images, labels, weights = archives
    extra = b"stack0"
    with tarfile.open(images, "a") as bundle:
        info = tarfile.TarInfo("positive/10003_2020-01-04T00-00-00.000Z_0.tif")
        info.size = len(extra)
        bundle.addfile(info, io.BytesIO(extra))
    source = ZenodoSource(images, labels, load_weight_table(weights))
    new_refs = [ref for ref in source.index() if ref.tile_id.startswith("10003_2020-01-04")]
    assert len(new_refs) == len(DEFAULT_BINS)
    assert {ref.group_id for ref in new_refs} == {"10003"}
    assert not any(ref.has_mask for ref in new_refs)
    dest = tmp_path / "ds"
    build_zenodo(images, labels, weights, dest, BuildSpec())
    group_splits: dict[str, set[str]] = {}
    for task, split, shard in _shard_dirs(dest):
        for row in read_rows(shard):
            if task == "classifier":
                group_splits.setdefault(row.group_id, set()).add(split)
            if split != "train":
                assert row.element == "id"
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert len(group_splits["10003"]) == 1


def test_build_rejects_weight_table_mismatch(
    archives: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """A spec weight_table_id that disagrees with the TOML fails."""
    images, labels, weights = archives
    spec = BuildSpec(weight_table_id="other-table")
    with pytest.raises(ValueError, match="weight_table_id"):
        build_zenodo(images, labels, weights, tmp_path / "ds", spec)


def test_cli_zenodo_build_and_bin_filter(archives: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """The CLI builds the dataset and honours repeatable --bin-id."""
    images, labels, weights = archives
    dest = tmp_path / "ds"
    code = main(
        [
            "dataset",
            "build",
            "--source",
            "zenodo",
            "--images-tar",
            str(images),
            "--labels-tar",
            str(labels),
            "--weights-path",
            str(weights),
            "--bin-id",
            "native10",
            "--bin-id",
            "gsd35",
            "--out",
            str(dest),
        ]
    )
    assert code == 0
    manifest = load_manifest(dest / "dataset.json")
    assert manifest.weight_table_id == "ap3200t-test"
    assert {item.bin_id for item in manifest.bins} == {"native10", "gsd35"}


def test_cli_zenodo_rejects_bad_args(tmp_path: Path) -> None:
    """Missing options, unknown and duplicate bin ids are parameter errors."""
    images, labels, weights = tmp_path / "i.tar", tmp_path / "l.tar", tmp_path / "w.toml"
    base = ["dataset", "build", "--source", "zenodo", "--out", str(tmp_path / "o")]
    assert main([*base]) != 0
    full = [
        *base[:-2],
        "--images-tar",
        str(images),
        "--labels-tar",
        str(labels),
        "--weights-path",
        str(weights),
        "--out",
        str(tmp_path / "o2"),
    ]
    assert main([*full, "--bin-id", "nope"]) != 0
    assert main([*full, "--bin-id", "elevation45"]) != 0
    assert main([*full, "--bin-id", "native10", "--bin-id", "native10"]) != 0
