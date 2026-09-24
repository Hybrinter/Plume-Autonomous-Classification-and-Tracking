"""Location splits stay together across seeds."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from tools.original_dataset_analysis.index import build_index
from tools.original_dataset_analysis.split import SplitRecipe, assign_location_splits


def _member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    """Append one in-memory member."""
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _write_archives(tmp_path: Path) -> tuple[Path, Path]:
    """Write a tiny image archive and a label archive sharing two sites."""
    images = tmp_path / "images.tar"
    labels = tmp_path / "labels.tar"
    with tarfile.open(images, "w") as archive:
        for name in (
            "positive/10_2019-01-01T00:00:00.000Z_0.tif",
            "negative/10_2019-02-01T00:00:00.000Z_0.tif",
            "positive/20_2019-01-01T00:00:00.000Z_0.tif",
            "negative/30_2019-01-01T00:00:00.000Z_0.tif",
        ):
            _member(archive, name, b"not-a-tiff")
    annotation = {
        "completions": [
            {
                "result": [
                    {
                        "type": "polygonlabels",
                        "value": {
                            "polygonlabels": ["smoke"],
                            "points": [[0, 0], [10, 0], [10, 10]],
                        },
                    }
                ]
            }
        ]
    }
    empty = {"completions": [{"result": []}]}
    with tarfile.open(labels, "w") as archive:
        _member(
            archive,
            "10_2019-01-01T00:00:00.000Z_0_features.json",
            json.dumps(annotation).encode(),
        )
        _member(
            archive,
            "20_2019-01-01T00:00:00.000Z_0_features.json",
            json.dumps(empty).encode(),
        )
    return images, labels


def test_shared_location_stays_together(tmp_path: Path) -> None:
    """Two stems that share a location id land in the same split for several seeds."""
    images, labels = _write_archives(tmp_path)
    index = build_index(images, labels)
    stems = {tile.stem: tile for tile in index.tiles}
    assert stems["10_2019-01-01T00:00:00.000Z_0"].positive is True
    assert stems["10_2019-01-01T00:00:00.000Z_0"].polygons is not None
    assert len(stems["10_2019-01-01T00:00:00.000Z_0"].polygons) == 1
    assert stems["20_2019-01-01T00:00:00.000Z_0"].polygons == ()
    assert stems["30_2019-01-01T00:00:00.000Z_0"].polygons is None
    for seed in (0, 1, 2):
        split = assign_location_splits(index, SplitRecipe(seed=seed))
        first = split.location_to_split["10"]
        for tile in index.tiles:
            if tile.location_id == "10":
                assert tile.stem in split.stems[first]
                assert split.location_to_split[tile.location_id] == first
