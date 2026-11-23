"""Zenodo 4250706 checksum manifest, download, and verification.

Contains:
  - DatasetFile / DatasetManifest: the pinned record schema.
  - load_dataset_manifest / file_md5 / verify_file: checksum plumbing.
  - download_file: verified HTTP fetch for one manifest entry.
  - main: CLI used by ``scripts/fetch_smoke_plume_dataset.py``.

Nothing downloads without ``--download`` and no pixel conversion happens here;
finished datasets are built by ``tools.ml_models.dataset``.
"""

from __future__ import annotations

import argparse
import hashlib
import tomllib
import urllib.request
from pathlib import Path

from pydantic import ConfigDict, TypeAdapter, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass


def _repo_root() -> Path:
    """Return the repository root that holds data/manifests."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "manifests").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[8]


REPO_ROOT = _repo_root()
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "zenodo_4250706.toml"
DEFAULT_RAW = REPO_ROOT / "data" / "raw"

# Fallback if the TOML omits indices: B2, B3, B4, B8 in the 13-band GeoTIFF.
DEFAULT_PACT_BAND_INDICES: tuple[int, int, int, int] = (1, 2, 3, 7)
DEFAULT_DN_SCALE = 10000.0


_SCHEMA = ConfigDict(extra="forbid")


@pydantic_dataclass(frozen=True, slots=True, config=_SCHEMA)
class DatasetFile:
    """One Zenodo file entry."""

    key: str
    size: int
    md5: str
    url: str


@pydantic_dataclass(frozen=True, slots=True, config=_SCHEMA)
class DatasetManifest:
    """Pinned Zenodo record plus file checksums."""

    record_id: int
    doi: str
    title: str
    citation: str
    files: tuple[DatasetFile, ...]
    pact_band_indices: tuple[int, ...] = DEFAULT_PACT_BAND_INDICES
    dn_scale: float = DEFAULT_DN_SCALE

    @field_validator("citation")
    @classmethod
    def _strip_citation(cls, value: str) -> str:
        """Strip leading and trailing whitespace from the citation string."""
        return value.strip()


def load_dataset_manifest(path: str | Path | None = None) -> DatasetManifest:
    """Parse the Zenodo checksum manifest.

    Args:
        path: TOML path. None uses ``data/manifests/zenodo_4250706.toml``.

    Returns:
        DatasetManifest: Record metadata and file list.

    Raises:
        OSError / tomllib.TOMLDecodeError: on a missing or malformed file.
        ValidationError: if a required field is missing or a key is unknown.
    """
    dest = Path(path) if path is not None else DEFAULT_MANIFEST
    data = tomllib.loads(dest.read_text(encoding="utf-8"))
    return TypeAdapter(DatasetManifest).validate_python(data)


def file_md5(path: str | Path) -> str:
    """Return the md5 hex digest of a file.

    Args:
        path: Filesystem path.

    Returns:
        str: Lowercase hex md5.
    """
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: str | Path, expected_md5: str, expected_size: int) -> bool:
    """Return True when size and md5 match.

    Args:
        path: Local file.
        expected_md5: Lowercase hex md5.
        expected_size: Byte size from the manifest.

    Returns:
        bool: True on a match. Missing files return False.
    """
    dest = Path(path)
    if not dest.is_file():
        return False
    if dest.stat().st_size != expected_size:
        return False
    return file_md5(dest) == expected_md5.lower()


def download_file(url: str, dest: Path, expected_md5: str, expected_size: int) -> None:
    """Download ``url`` to ``dest`` and verify checksum.

    Args:
        url: HTTP(S) source.
        dest: Local destination path.
        expected_md5: Manifest md5.
        expected_size: Manifest size.

    Raises:
        ValueError: If the downloaded file fails size or md5.
        OSError: On a network or write failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 — URL comes from the pinned manifest
    if not verify_file(dest, expected_md5, expected_size):
        raise ValueError(f"checksum mismatch after download: {dest}")


def _status_line(entry: DatasetFile, raw_dir: Path) -> str:
    """Return a one-line presence/checksum status for a manifest file."""
    path = raw_dir / entry.key
    ok = verify_file(path, entry.md5, entry.size)
    state = "ok" if ok else ("missing" if not path.is_file() else "mismatch")
    return f"{entry.key}: {state}"


def main(argv: list[str] | None = None) -> int:
    """CLI: print citation and checksum status; download only on request.

    Args:
        argv: Argument list without the program name.

    Returns:
        int: 0 on success. 1 on checksum failure.
    """
    parser = argparse.ArgumentParser(prog="fetch_smoke_plume_dataset")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="checksum manifest TOML",
    )
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    parser.add_argument(
        "--download",
        action="store_true",
        help="fetch missing or mismatched files from Zenodo",
    )
    args = parser.parse_args(argv)

    manifest = load_dataset_manifest(args.manifest)
    print(manifest.citation)
    print(f"DOI: {manifest.doi}")
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest.files:
        print(_status_line(entry, raw_dir))
        if not args.download:
            continue
        dest = raw_dir / entry.key
        if verify_file(dest, entry.md5, entry.size):
            continue
        print(f"downloading {entry.key} ...")
        download_file(entry.url, dest, entry.md5, entry.size)
        print(_status_line(entry, raw_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
