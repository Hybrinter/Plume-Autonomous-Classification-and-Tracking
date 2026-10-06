"""Tests for the Zenodo archive index and GeoTIFF streaming."""

import io
import tarfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.dataset.sources.zenodo.archive import (
    TileRef,
    acquired_at_utc_of,
    build_index,
    iter_stacks,
    location_id_of,
    observation_metadata,
    to_native_stack,
)


def test_index_reads_class_dirs_and_annotations(
    archives: tuple[Path, Path, Path], stems: tuple[str, str, str]
) -> None:
    """Positive comes from the class dir; polygons from the label archive."""
    images, labels, _weights = archives
    positive, negative, bare = stems
    index = build_index(images, labels)
    by_stem = index.by_stem()
    assert [tile.stem for tile in index.tiles] == list(stems)
    assert by_stem[positive].positive and not by_stem[negative].positive
    assert len(by_stem[positive].polygons or ()) == 1
    assert len(by_stem[negative].polygons or ()) == 1
    assert by_stem[bare].polygons is None
    assert {tile.location_id for tile in index.tiles} == {"10003", "10004", "10005"}


def test_image_member_outside_class_dirs_is_rejected(
    tmp_path: Path, archives: tuple[Path, Path, Path]
) -> None:
    """An image not under exactly one class directory cannot index."""
    _images, labels, _weights = archives
    for name in ("other/x_0.tif", "positive/negative/y_0.tif"):
        stray = tmp_path / "stray.tar"
        with tarfile.open(stray, "w") as bundle:
            info = tarfile.TarInfo(name)
            info.size = 1
            bundle.addfile(info, io.BytesIO(b"x"))
        with pytest.raises(ValueError, match="positive"):
            build_index(stray, labels)
        stray.unlink()


def test_iter_stacks_is_one_forward_pass(
    archives: tuple[Path, Path, Path],
    stems: tuple[str, str, str],
    stub_stacks: list[str],
) -> None:
    """Each image member decodes exactly once, in archive order."""
    images, labels, _weights = archives
    index = build_index(images, labels)
    seen = [tile.stem for tile, _stack, _desc in iter_stacks(images, index.tiles)]
    assert seen == list(stems)
    assert stub_stacks == ["stack0", "stack1", "stack2"]


def test_missing_member_is_reported(archives: tuple[Path, Path, Path]) -> None:
    """A requested member absent from the archive raises FileNotFoundError."""
    images, labels, _weights = archives
    index = build_index(images, labels)
    phantom = replace(index.tiles[0], member_name="positive/ghost_0.tif")
    with pytest.raises(FileNotFoundError):
        list(iter_stacks(images, (*index.tiles, phantom)))


def test_to_native_stack_fits_both_axes() -> None:
    """A 119 by 121 source pads H and crops W onto the 120 square."""
    stack = np.zeros((2, 119, 121), dtype=np.float32)
    stack[:, 5, :] = 7.0
    fitted = to_native_stack(stack)
    assert fitted.shape == (2, 120, 120)
    np.testing.assert_array_equal(fitted[:, :119, :120], stack[:, :, :120])
    np.testing.assert_array_equal(fitted[:, 119, :], fitted[:, 118, :])
    with pytest.raises(ValueError, match="120"):
        to_native_stack(np.zeros((2, 100, 100), dtype=np.float32))
    with pytest.raises(ValueError, match="C, H, W"):
        to_native_stack(np.zeros((4, 4), dtype=np.float32))


def test_location_id_is_leading_token() -> None:
    """The location token is the leading underscore run."""
    assert location_id_of("10003_2020-01-01_0") == "10003"
    assert location_id_of("single") == "single"


@pytest.mark.parametrize(
    "stem,timestamp",
    [
        ("10003_2019-01-21T10:56:41.330Z_0", "2019-01-21T10:56:41.330Z"),
        ("10003_2020-02-29T00:00:00Z_2", "2020-02-29T00:00:00Z"),
        ("10003_2020-02-30T00:00:00Z_0", None),
        ("10003_2020-01-01T00-00-00.000Z_0", None),
        ("10003_2020-01-01T00:00:00+01:00_0", None),
        ("10003_2020-01-01_0", None),
        ("prefix_2020-01-01T00:00:00Z_suffix", None),
        ("bare", None),
    ],
)
def test_only_documented_valid_utc_stems_provide_acquisition_time(
    stem: str,
    timestamp: str | None,
) -> None:
    assert acquired_at_utc_of(stem) == timestamp


def test_zenodo_observation_and_source_annotation_are_not_derived_from_label() -> None:
    polygon = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float64)
    base = TileRef(
        stem="10003_2019-01-21T10:56:41.330Z_0",
        location_id="10003",
        positive=False,
        member_name="negative/source.tif",
        polygons=(polygon,),
        acquired_at_utc="2019-01-21T10:56:41.330Z",
        annotation_ref="annotations/actual-source.json",
    )
    metadata = observation_metadata(base)
    assert metadata.observation_id == base.stem
    assert metadata.acquired_at_utc == base.acquired_at_utc
    assert metadata.conditions == ()
    assert metadata.annotation_source == "annotations/actual-source.json"
    assert metadata.source_annotation_state == "NONEMPTY"
    explicit_empty = observation_metadata(replace(base, polygons=()))
    assert explicit_empty.source_annotation_state == "EXPLICIT_EMPTY"
    missing = observation_metadata(replace(base, polygons=None, annotation_ref=None))
    assert missing.source_annotation_state == "MISSING"
    assert missing.annotation_source is None
