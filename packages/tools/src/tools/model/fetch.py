"""Zenodo 4250706 fetch, checksum verify, and optional 4-band preprocess.

Default invocation does not download. Pass ``--download`` to fetch missing or
mismatched files into ``data/raw/``. ``--preprocess`` converts GeoTIFF or packed
numpy planes into the PACT BLUE/GREEN/RED/NIR ``normalize_dn``-style [0, 1]
domain under ``data/processed/``.

Contains:
  - DatasetManifest / DatasetFile: checksummed Zenodo file list.
  - load_dataset_manifest: parse data/manifests/zenodo_4250706.toml.
  - file_md5: md5 hex digest of a path.
  - verify_file: size + md5 check.
  - select_pact_bands / to_model_domain: 13-band -> (4, H, W) float32 [0, 1].
  - download_file: urllib fetch with checksum.
  - preprocess_planes / preprocess_tree: write packed numpy batches.
  - main: CLI used by scripts/fetch_smoke_plume_dataset.py.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import argparse
import hashlib
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np


def _repo_root() -> Path:
    """Return the repository root that holds data/manifests."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "manifests").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[5]


REPO_ROOT = _repo_root()
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "zenodo_4250706.toml"
DEFAULT_RAW = REPO_ROOT / "data" / "raw"
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"

# Fallback if the TOML omits indices: B2, B3, B4, B8 in the 13-band GeoTIFF.
DEFAULT_PACT_BAND_INDICES: tuple[int, int, int, int] = (1, 2, 3, 7)
DEFAULT_DN_SCALE = 10000.0


@dataclass(frozen=True, slots=True)
class DatasetFile:
    """One Zenodo file entry."""

    key: str
    size: int
    md5: str
    url: str


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Pinned Zenodo record plus file checksums."""

    record_id: int
    doi: str
    title: str
    citation: str
    pact_band_indices: tuple[int, ...]
    dn_scale: float
    files: tuple[DatasetFile, ...]


def load_dataset_manifest(path: str | Path | None = None) -> DatasetManifest:
    """Parse the Zenodo checksum manifest.

    Args:
        path: TOML path. None uses ``data/manifests/zenodo_4250706.toml``.

    Returns:
        DatasetManifest: Record metadata and file list.

    Raises:
        OSError / tomllib.TOMLDecodeError / KeyError: on a missing or malformed file.
    """
    dest = Path(path) if path is not None else DEFAULT_MANIFEST
    data = tomllib.loads(dest.read_text(encoding="utf-8"))
    files = tuple(
        DatasetFile(
            key=str(item["key"]),
            size=int(item["size"]),
            md5=str(item["md5"]),
            url=str(item["url"]),
        )
        for item in data["files"]
    )
    indices = tuple(int(v) for v in data.get("pact_band_indices", DEFAULT_PACT_BAND_INDICES))
    return DatasetManifest(
        record_id=int(data["record_id"]),
        doi=str(data["doi"]),
        title=str(data["title"]),
        citation=str(data["citation"]).strip(),
        pact_band_indices=indices,
        dn_scale=float(data.get("dn_scale", DEFAULT_DN_SCALE)),
        files=files,
    )


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


def select_pact_bands(
    planes: np.ndarray,
    indices: tuple[int, ...] = DEFAULT_PACT_BAND_INDICES,
) -> np.ndarray:
    """Take BLUE/GREEN/RED/NIR planes from a 13-band stack.

    Args:
        planes: np.ndarray[..., C, H, W] or (C, H, W) with C >= max(indices)+1.
        indices: Band indices in the source stack.

    Returns:
        np.ndarray: Same rank as input with C replaced by len(indices).
    """
    arr = np.asarray(planes)
    if arr.ndim < 3:
        raise ValueError("planes must have at least rank 3 (C, H, W)")
    return np.take(arr, indices, axis=-3)


def _resize_hw(planes: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize of (C, H, W) to (C, height, width)."""
    _c, src_h, src_w = planes.shape
    row = (np.arange(height) * src_h // height).astype(np.int64)
    col = (np.arange(width) * src_w // width).astype(np.int64)
    return planes[:, row[:, None], col]


def to_model_domain(
    planes: np.ndarray,
    height: int,
    width: int,
    indices: tuple[int, ...] = DEFAULT_PACT_BAND_INDICES,
    dn_scale: float = DEFAULT_DN_SCALE,
) -> np.ndarray:
    """Map a 13-band (or 4-band) stack to (4, H, W) float32 in [0, 1].

    Args:
        planes: np.ndarray[float, (C, h, w)] source bands.
        height: Output height.
        width: Output width.
        indices: Source indices when C > 4. Ignored when C == 4.
        dn_scale: Divide then clip. Sentinel-2 L2A uses 10000. Already-normalized
            planes use 1.0.

    Returns:
        np.ndarray[float32, (4, height, width)] in [0, 1].
    """
    arr = np.asarray(planes, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected (C, H, W); got {arr.shape}")
    if arr.shape[0] == 4:
        selected = arr
    else:
        selected = select_pact_bands(arr, indices).astype(np.float32)
    scaled = selected / float(dn_scale)
    clipped = np.clip(scaled, 0.0, 1.0)
    if clipped.shape[-2:] != (height, width):
        clipped = _resize_hw(clipped, height, width)
    return cast(np.ndarray, clipped.astype(np.float32))


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


def preprocess_planes(
    planes: np.ndarray,
    height: int = 256,
    width: int = 256,
    indices: tuple[int, ...] = DEFAULT_PACT_BAND_INDICES,
    dn_scale: float = DEFAULT_DN_SCALE,
) -> np.ndarray:
    """Convert one source stack to a PACT model-domain tensor.

    Args:
        planes: (C, h, w) source.
        height: Output height (InferenceConfig default 256).
        width: Output width.
        indices: Band indices when C != 4.
        dn_scale: Source DN scale.

    Returns:
        np.ndarray[float32, (4, height, width)].
    """
    return to_model_domain(planes, height, width, indices=indices, dn_scale=dn_scale)


def preprocess_tree(
    source_dir: Path,
    dest_dir: Path,
    height: int,
    width: int,
    indices: tuple[int, ...],
    dn_scale: float,
    limit: int = 0,
) -> int:
    """Preprocess ``*.npy`` stacks under source_dir into dest_dir/images.npy.

    Args:
        source_dir: Directory of (C, h, w) ``.npy`` files.
        dest_dir: Output directory.
        height: Output height.
        width: Output width.
        indices: Band indices.
        dn_scale: Source DN scale.
        limit: Max files (0 means no limit).

    Returns:
        int: Number of samples written.

    Notes:
        GeoTIFF conversion requires rasterio and runs only when ``.tif`` files
        are present and the extra is installed.
    """
    npy_files = sorted(source_dir.glob("*.npy"))
    tif_files = sorted(source_dir.glob("*.tif")) + sorted(source_dir.glob("*.tiff"))
    stacks: list[np.ndarray] = []
    for path in npy_files:
        stacks.append(np.load(path))
        if limit > 0 and len(stacks) >= limit:
            break
    if len(stacks) < (limit or 10**9) and tif_files:
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError(
                "rasterio is required to preprocess GeoTIFF files; "
                "convert to .npy or install rasterio"
            ) from exc
        for path in tif_files:
            with rasterio.open(path) as src:
                planes = src.read().astype(np.float32)
            stacks.append(planes)
            if limit > 0 and len(stacks) >= limit:
                break
    if not stacks:
        raise FileNotFoundError(f"no .npy or .tif stacks in {source_dir}")
    images = np.stack(
        [
            preprocess_planes(item, height=height, width=width, indices=indices, dn_scale=dn_scale)
            for item in stacks
        ],
        axis=0,
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    np.save(dest_dir / "images.npy", images)
    return int(images.shape[0])


def _status_line(entry: DatasetFile, raw_dir: Path) -> str:
    """Return a one-line presence/checksum status for a manifest file."""
    path = raw_dir / entry.key
    ok = verify_file(path, entry.md5, entry.size)
    state = "ok" if ok else ("missing" if not path.is_file() else "mismatch")
    return f"{entry.key}: {state}"


def main(argv: list[str] | None = None) -> int:
    """CLI: print citation, optionally download, optionally preprocess.

    Args:
        argv: Argument list without the program name.

    Returns:
        int: 0 on success. 1 on checksum or preprocess failure.
    """
    parser = argparse.ArgumentParser(prog="fetch_smoke_plume_dataset")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="checksum manifest TOML",
    )
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED))
    parser.add_argument(
        "--download",
        action="store_true",
        help="fetch missing or mismatched files from Zenodo",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="write 4-band PACT tensors under --processed-dir",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="max preprocess samples (0 = all)")
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
    if args.preprocess:
        count = preprocess_tree(
            raw_dir,
            Path(args.processed_dir),
            height=args.height,
            width=args.width,
            indices=tuple(manifest.pact_band_indices),
            dn_scale=manifest.dn_scale,
            limit=args.limit,
        )
        print(f"preprocessed {count} sample(s) -> {args.processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
