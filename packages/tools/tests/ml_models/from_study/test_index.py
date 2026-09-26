"""Image stacks come from one forward pass of the archive."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path
from typing import IO, Literal

import numpy as np
import pytest
from tools.ml_models.data.zenodo import TileRef, iter_stacks


def _member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    """Append one in-memory member."""
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _tile(name: str) -> TileRef:
    """Return a tile that points at ``name`` inside the archive."""
    return TileRef(
        stem=Path(name).stem,
        location_id="10",
        positive="positive" in Path(name).parts,
        member_name=name,
        polygons=None,
    )


def _archive(tmp_path: Path, names: tuple[str, ...], *, gzip: bool) -> Path:
    """Write a tar whose members are named ``names``."""
    path = tmp_path / "images.tar"
    archive = tarfile.open(path, "w:gz") if gzip else tarfile.open(path, "w")
    with archive:
        for index, name in enumerate(names):
            _member(archive, name, f"payload-{index}".encode())
    return path


class _Dataset:
    """Stand-in for a one-band rasterio dataset."""

    descriptions = ("B2 blue", None)

    def read(self) -> np.ndarray:
        """Return a constant uint16 plane."""
        return np.ones((1, 2, 2), dtype=np.uint16)

    def __enter__(self) -> _Dataset:
        """Return this dataset."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the stand-in."""
        return None


class _Rasterio:
    """Stand-in for the rasterio module."""

    def open(self, _source: object) -> _Dataset:
        """Return the stand-in dataset."""
        return _Dataset()


def _install_rasterio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``import rasterio`` at the test double."""
    monkeypatch.setitem(sys.modules, "rasterio", _Rasterio())


@pytest.mark.parametrize("gzip", [False, True])
def test_iter_stacks_opens_the_archive_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gzip: bool,
) -> None:
    """Two tiles share one stream open, in archive order, using the member object."""
    first = "positive/10_a.tif"
    second = "negative/10_b.tif"
    path = _archive(tmp_path, (first, "notes.txt", second), gzip=gzip)
    _install_rasterio(monkeypatch)
    opens: list[str] = []
    looked_up_by_name: list[str] = []
    real_open = tarfile.open
    real_extract = tarfile.TarFile.extractfile

    def _counting_open(name: str | Path, mode: Literal["r|*"] = "r|*") -> tarfile.TarFile:
        opens.append(mode)
        return real_open(name, mode)

    def _guard_extract(
        self: tarfile.TarFile,
        member: tarfile.TarInfo | str,
    ) -> IO[bytes] | None:
        if isinstance(member, str):
            looked_up_by_name.append(member)
        return real_extract(self, member)

    monkeypatch.setattr(tarfile, "open", _counting_open)
    monkeypatch.setattr(tarfile.TarFile, "extractfile", _guard_extract)

    stacks = list(iter_stacks(path, (_tile(second), _tile(first))))

    assert opens == ["r|*"]
    assert looked_up_by_name == []
    assert [item[0].member_name for item in stacks] == [first, second]
    assert stacks[0][1].shape == (1, 2, 2)
    assert stacks[0][1].dtype == np.float32
    assert stacks[0][2] == ("B2 blue", "")


def test_iter_stacks_reports_a_missing_member(tmp_path: Path) -> None:
    """A requested member that the stream never reaches is an error."""
    path = _archive(tmp_path, ("positive/10_a.tif",), gzip=False)
    missing = _tile("positive/absent.tif")
    with pytest.raises(FileNotFoundError, match="absent.tif"):
        list(iter_stacks(path, (missing,)))


def test_empty_tile_list_does_not_open_the_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No tiles means no archive open."""
    path = _archive(tmp_path, ("positive/10_a.tif",), gzip=False)
    opens = 0
    real_open = tarfile.open

    def _counting_open(name: str | Path, mode: Literal["r|*"] = "r|*") -> tarfile.TarFile:
        nonlocal opens
        opens += 1
        return real_open(name, mode)

    monkeypatch.setattr(tarfile, "open", _counting_open)
    assert list(iter_stacks(path, ())) == []
    assert opens == 0


def test_missing_archive_raises_before_iteration(tmp_path: Path) -> None:
    """A missing archive fails when the iterator is constructed."""
    missing = tmp_path / "absent.tar"
    with pytest.raises(FileNotFoundError, match="absent.tar"):
        iter_stacks(missing, (_tile("positive/10_a.tif"),))
