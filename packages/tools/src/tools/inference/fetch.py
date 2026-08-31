"""Zenodo 4250706 fetch, checksum verify, and 4-band preprocess.

Default invocation does not download. Pass ``--download`` to fetch missing or
mismatched files into ``data/raw/``. ``--preprocess`` reads the Zenodo tarballs
and writes processed packs (images, masks, labels, splits, dataset hash) under
``data/processed/``.

Preprocessing reads the archives as streams rather than extracting them. Two
properties of the corpus force that. Filenames embed an ISO timestamp with
colons, which Windows rejects as a path character, and the images expand to
about 8 GB that nothing later needs on disk.

The corpus supports the two models unequally, so preprocessing writes two
packs. All 21,350 images carry a plume-presence label in their class
directory, but polygon masks cover only 1,437 of them. The classifier pack
holds the whole corpus; the segmentor pack holds the annotated subset.

Contains:
  - DatasetManifest / DatasetFile: checksummed Zenodo file list.
  - load_dataset_manifest: parse data/manifests/zenodo_4250706.toml.
  - file_md5: md5 hex digest of a path.
  - verify_file: size and md5 check.
  - select_pact_bands / to_model_domain: 13-band -> (4, H, W) float32 [0, 1].
  - download_file: urllib fetch with checksum.
  - preprocess_planes: 13-band or 4-band stack to model domain.
  - ZenodoIndex / read_annotation_archive / index_image_archive.
  - preprocess_zenodo_archives: stream both archives into two packs.
  - main: CLI used by scripts/fetch_smoke_plume_dataset.py.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
from pydantic import ConfigDict, TypeAdapter, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tools.inference.annotations import annotation_stem, parse_polygons, rasterize_polygons
from tools.inference.split import (
    DatasetMeta,
    SplitRecipe,
    assign_splits,
    compute_dataset_hash,
    load_split_recipe,
    write_dataset_meta,
    write_splits,
)

_GEOTIFF_SUFFIXES = frozenset({".tif", ".tiff"})
_POSITIVE_DIR = "positive"


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


@dataclass(frozen=True, slots=True)
class ZenodoIndex:
    """Sorted contents of the Zenodo image archive.

    Attributes:
        stems: Image stems in sorted order. Split assignment uses this order.
        positive_stems: Stems the corpus files under ``positive``.
        annotated_stems: Stems that have a segmentation annotation.
    """

    stems: tuple[str, ...]
    positive_stems: frozenset[str]
    annotated_stems: frozenset[str]


def _member_stem(name: str) -> str:
    """Return the filename stem of a tar member path."""
    return PurePosixPath(name).stem


def read_annotation_archive(labels_archive: str | Path) -> dict[str, tuple[np.ndarray, ...]]:
    """Return percentage-space polygons for every annotated stem.

    Args:
        labels_archive: ``segmentation_labels.tar.gz`` path.

    Returns:
        dict[str, tuple[np.ndarray, ...]]: Image stem to its polygons. A stem
        annotated as having no plume maps to an empty tuple, which is a
        negative rather than a missing entry.

    Raises:
        FileNotFoundError: If the archive is missing.
        tarfile.TarError: If the archive is malformed.

    Notes:
        Annotations are read straight out of the archive. Corpus filenames
        embed an ISO timestamp containing colons, which Windows rejects as a
        path character, so extracting the archive to disk is not portable.
    """
    src = Path(labels_archive)
    if not src.is_file():
        raise FileNotFoundError(f"archive not found: {src}")
    polygons: dict[str, tuple[np.ndarray, ...]] = {}
    with tarfile.open(src, "r|gz") as handle:
        for member in handle:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            payload = handle.extractfile(member)
            if payload is None:
                continue
            stem = annotation_stem(PurePosixPath(member.name).name)
            polygons[stem] = parse_polygons(json.loads(payload.read().decode("utf-8")))
    return polygons


def index_image_archive(
    images_archive: str | Path,
    annotated_stems: frozenset[str],
) -> ZenodoIndex:
    """Walk the image archive and record stems and their class directory.

    Args:
        images_archive: ``images.tar.gz`` path.
        annotated_stems: Stems that have a segmentation annotation.

    Returns:
        ZenodoIndex: Sorted stems plus the positive and annotated subsets.

    Raises:
        FileNotFoundError: If the archive is missing.
        tarfile.TarError: If the archive is malformed.

    Notes:
        A gzip stream can only be read forward, so the archive is walked once
        here to learn its size and class labels, then a second time to fill the
        pack. Preallocating the pack is what makes a corpus larger than memory
        writable.
    """
    src = Path(images_archive)
    if not src.is_file():
        raise FileNotFoundError(f"archive not found: {src}")
    stems: list[str] = []
    positive: set[str] = set()
    with tarfile.open(src, "r|gz") as handle:
        for member in handle:
            if not member.isfile():
                continue
            if PurePosixPath(member.name).suffix.lower() not in _GEOTIFF_SUFFIXES:
                continue
            stem = _member_stem(member.name)
            stems.append(stem)
            if PurePosixPath(member.name).parent.name == _POSITIVE_DIR:
                positive.add(stem)
    ordered = tuple(sorted(stems))
    return ZenodoIndex(
        stems=ordered,
        positive_stems=frozenset(positive),
        annotated_stems=frozenset(annotated_stems & set(ordered)),
    )


def _read_geotiff_bytes(payload: bytes) -> np.ndarray:
    """Return the (C, h, w) float32 band stack of an in-memory GeoTIFF."""
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required to preprocess the Zenodo GeoTIFF corpus; "
            "install the pact-tools 'data' extra"
        ) from exc
    with rasterio.io.MemoryFile(payload) as memfile, memfile.open() as handle:
        return np.asarray(handle.read(), dtype=np.float32)


def _finalize_pack(
    dest: Path,
    n: int,
    height: int,
    width: int,
    in_channels: int,
    recipe: SplitRecipe,
    source_doi: str,
) -> None:
    """Write splits and identity for a pack whose tensors are already on disk."""
    write_splits(dest / "splits.json", assign_splits(n, recipe))
    meta = DatasetMeta(
        dataset_hash=compute_dataset_hash(dest),
        source_doi=source_doi,
        n=n,
        height=height,
        width=width,
        in_channels=in_channels,
    )
    write_dataset_meta(dest / "dataset.json", meta)


def preprocess_zenodo_archives(
    images_archive: str | Path,
    labels_archive: str | Path,
    classifier_dir: str | Path | None,
    segmentor_dir: str | Path | None,
    height: int = 256,
    width: int = 256,
    indices: tuple[int, ...] = DEFAULT_PACT_BAND_INDICES,
    dn_scale: float = DEFAULT_DN_SCALE,
    recipe_path: str | Path | None = None,
    source_doi: str = "",
    limit: int = 0,
) -> tuple[int, int]:
    """Stream the Zenodo archives into a classifier pack and a segmentor pack.

    Args:
        images_archive: ``images.tar.gz`` path.
        labels_archive: ``segmentation_labels.tar.gz`` path.
        classifier_dir: Output directory for the whole-corpus pack, or None to
            skip it.
        segmentor_dir: Output directory for the annotated-subset pack, or None
            to skip it.
        height: Output height.
        width: Output width.
        indices: Band indices into the 13-band source stack.
        dn_scale: Source DN scale.
        recipe_path: Split-recipe TOML. None uses the committed default.
        source_doi: DOI recorded in each ``dataset.json``.
        limit: Max classifier samples (0 means all). The segmentor pack is
            capped to the annotated stems within that prefix.

    Returns:
        tuple[int, int]: Sample counts written to the classifier pack and the
        segmentor pack.

    Raises:
        FileNotFoundError: If an archive is missing.
        ImportError: If rasterio is not installed.
        ValueError: If no samples are selected for a requested pack.

    Notes:
        The two packs cover different populations on purpose. Plume presence is
        labelled for all 21,350 images by their class directory, but polygon
        masks exist for only 1,437 of them. Training a segmentor on the whole
        corpus would present the unannotated majority as empty masks and teach
        it to predict nothing, so the segmentor pack holds the annotated subset
        alone.
    """
    polygons = read_annotation_archive(labels_archive)
    index = index_image_archive(images_archive, frozenset(polygons))
    stems = index.stems[:limit] if limit > 0 else index.stems
    selected = set(stems)
    seg_stems = tuple(stem for stem in stems if stem in index.annotated_stems)
    if classifier_dir is not None and not stems:
        raise ValueError("no images selected for the classifier pack")
    if segmentor_dir is not None and not seg_stems:
        raise ValueError("no annotated images selected for the segmentor pack")

    recipe = load_split_recipe(recipe_path)
    cls_writer = (
        _PackWriter(Path(classifier_dir), stems, height, width, len(indices))
        if classifier_dir is not None
        else None
    )
    seg_writer = (
        _PackWriter(Path(segmentor_dir), seg_stems, height, width, len(indices))
        if segmentor_dir is not None
        else None
    )

    with tarfile.open(Path(images_archive), "r|gz") as handle:
        for member in handle:
            if not member.isfile():
                continue
            if PurePosixPath(member.name).suffix.lower() not in _GEOTIFF_SUFFIXES:
                continue
            stem = _member_stem(member.name)
            if stem not in selected:
                continue
            payload = handle.extractfile(member)
            if payload is None:
                continue
            planes = _read_geotiff_bytes(payload.read())
            image = preprocess_planes(
                planes, height=height, width=width, indices=indices, dn_scale=dn_scale
            )
            label = 1.0 if stem in index.positive_stems else 0.0
            mask = rasterize_polygons(polygons.get(stem, ()), height, width)[np.newaxis, ...]
            if cls_writer is not None:
                cls_writer.write(stem, image, mask, label)
            if seg_writer is not None and stem in index.annotated_stems:
                seg_writer.write(stem, image, mask, label)

    written_cls = 0
    written_seg = 0
    if cls_writer is not None:
        written_cls = cls_writer.close()
        _finalize_pack(
            cls_writer.dest, written_cls, height, width, len(indices), recipe, source_doi
        )
    if seg_writer is not None:
        written_seg = seg_writer.close()
        _finalize_pack(
            seg_writer.dest, written_seg, height, width, len(indices), recipe, source_doi
        )
    return written_cls, written_seg


class _PackWriter:
    """Fill preallocated pack tensors on disk, one sample at a time."""

    def __init__(
        self,
        dest: Path,
        stems: tuple[str, ...],
        height: int,
        width: int,
        in_channels: int,
    ) -> None:
        """Preallocate ``images.npy``, ``masks.npy``, and ``labels.npy``."""
        dest.mkdir(parents=True, exist_ok=True)
        self.dest = dest
        self._slot = {stem: position for position, stem in enumerate(stems)}
        self._seen = 0
        n = len(stems)
        self._images = np.lib.format.open_memmap(
            dest / "images.npy", mode="w+", dtype=np.float32, shape=(n, in_channels, height, width)
        )
        self._masks = np.lib.format.open_memmap(
            dest / "masks.npy", mode="w+", dtype=np.float32, shape=(n, 1, height, width)
        )
        self._labels = np.lib.format.open_memmap(
            dest / "labels.npy", mode="w+", dtype=np.float32, shape=(n, 1)
        )
        (dest / "stems.json").write_text(json.dumps(list(stems), indent=0), encoding="utf-8")

    def write(self, stem: str, image: np.ndarray, mask: np.ndarray, label: float) -> None:
        """Store one sample at the slot its sorted stem reserved."""
        position = self._slot.get(stem)
        if position is None:
            return
        self._images[position] = image
        self._masks[position] = mask
        self._labels[position, 0] = label
        self._seen += 1

    def close(self) -> int:
        """Flush the memmaps and return the number of samples written."""
        self._images.flush()
        self._masks.flush()
        self._labels.flush()
        return self._seen


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
        "--classifier-dir",
        default=None,
        help="classifier pack directory (default: <processed-dir>/classifier)",
    )
    parser.add_argument(
        "--segmentor-dir",
        default=None,
        help="segmentor pack directory (default: <processed-dir>/segmentor)",
    )
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
    parser.add_argument(
        "--split-recipe",
        default=None,
        help="split recipe TOML (default: data/manifests/zenodo_4250706_splits.toml)",
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
    if args.preprocess:
        processed = Path(args.processed_dir)
        classifier_dir = (
            Path(args.classifier_dir) if args.classifier_dir else processed / "classifier"
        )
        segmentor_dir = Path(args.segmentor_dir) if args.segmentor_dir else processed / "segmentor"
        n_cls, n_seg = preprocess_zenodo_archives(
            raw_dir / "images.tar.gz",
            raw_dir / "segmentation_labels.tar.gz",
            classifier_dir,
            segmentor_dir,
            height=args.height,
            width=args.width,
            indices=tuple(manifest.pact_band_indices),
            dn_scale=manifest.dn_scale,
            recipe_path=args.split_recipe,
            source_doi=manifest.doi,
            limit=args.limit,
        )
        print(f"classifier pack: {n_cls} sample(s) -> {classifier_dir}")
        print(f"segmentor pack: {n_seg} sample(s) -> {segmentor_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
