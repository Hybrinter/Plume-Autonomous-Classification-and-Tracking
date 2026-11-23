"""Finished-dataset manifest and content hash.

Contains:
  - SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS, ShardCount, DatasetManifest.
  - compute_dataset_hash, write_manifest, load_manifest.

New builds write schema 3; schemas 2 and 3 read. The hash covers every
file under the dataset directory except ``dataset.json``. Paths are
relative, POSIX, and sorted. Each file is digested in 8 MiB chunks.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Self

from pydantic import ConfigDict, Field, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tools.ml_models.dataset.augment import AugmentRecipe
from tools.ml_models.dataset.split import SplitRecipe

SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = (2, 3)
_SCHEMA = ConfigDict(extra="forbid")
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_MANIFEST_NAME = "dataset.json"


@pydantic_dataclass(frozen=True, slots=True, config=_SCHEMA)
class BinRecord:
    """One bin row inside ``dataset.json``.

    Attributes:
        bin_id: Stable bin name.
        lateral_m: Nominal lateral GSD.
        along_m: Nominal along-track GSD.
        elevation_deg: Gimbal elevation, or None.
    """

    bin_id: str
    lateral_m: float
    along_m: float
    elevation_deg: float | None = None


@pydantic_dataclass(frozen=True, slots=True, config=_SCHEMA)
class ShardCount:
    """Row count for one task, split, and spatial size.

    Attributes:
        task: ``classifier`` or ``segmentor``.
        split: ``train``, ``val``, or ``test``.
        height: Tile H.
        width: Tile W.
        n: Rows in the shard, including train augmentations.
        n_positive: Rows whose label is at least 0.5.
    """

    task: str
    split: str
    height: int
    width: int
    n: int
    n_positive: int

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        """Reject a non-positive size or a positive count above ``n``."""
        if self.n < 1:
            raise ValueError(f"shard n must be >= 1; got {self.n}")
        if self.height < 1 or self.width < 1:
            raise ValueError(f"shard size must be >= 1; got {self.height}x{self.width}")
        if self.n_positive < 0 or self.n_positive > self.n:
            raise ValueError(f"n_positive must lie in 0..n; got {self.n_positive} of {self.n}")
        return self


@pydantic_dataclass(frozen=True, slots=True, config=_SCHEMA)
class DatasetManifest:
    """Identity file stored as ``dataset.json``.

    ``schema_version`` uses ``Field(strict=True)``; a ``Field`` assignment
    counts as a dataclass default, so it is declared after the required
    fields.

    Attributes:
        source: Source name.
        source_ref: DOI or other provenance string.
        weight_table_id: Class-weight table identifier.
        band_names: Channel names.
        norm: Always ``unit``.
        image_dtype: Always ``float32``.
        gsd_reference_m: Reference used to encode model GSD.
        split: Split recipe that produced the row assignment.
        augment: Augment recipe requested for train rows.
        bins: Named GSD bins. Empty when the source has none.
        shards: One count record per written shard.
        gsd_lateral_min_m: Minimum stored lateral GSD.
        gsd_lateral_max_m: Maximum stored lateral GSD.
        gsd_along_min_m: Minimum stored along-track GSD.
        gsd_along_max_m: Maximum stored along-track GSD.
        dataset_hash: Lowercase SHA-256 over the shard files.
        schema_version: Manifest schema. Written as JSON key ``schema``.
    """

    source: str
    source_ref: str
    weight_table_id: str
    band_names: tuple[str, ...]
    norm: str
    image_dtype: str
    gsd_reference_m: float
    split: SplitRecipe
    augment: AugmentRecipe
    bins: tuple[BinRecord, ...]
    shards: tuple[ShardCount, ...]
    gsd_lateral_min_m: float
    gsd_lateral_max_m: float
    gsd_along_min_m: float
    gsd_along_max_m: float
    dataset_hash: str
    schema_version: int = Field(strict=True)

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        """Reject a bad schema, norm, dtype, band list, or GSD range."""
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported dataset schema {self.schema_version}; "
                f"rebuild the dataset (current schema {SCHEMA_VERSION})"
            )
        if self.norm != "unit":
            raise ValueError(f"norm must be 'unit'; got {self.norm!r}")
        if self.image_dtype != "float32":
            raise ValueError(f"image_dtype must be 'float32'; got {self.image_dtype!r}")
        if len(self.band_names) < 1:
            raise ValueError("band_names must be non-empty")
        if len(self.shards) < 1:
            raise ValueError("shards must be non-empty")
        _require_gsd_range(self.gsd_lateral_min_m, self.gsd_lateral_max_m, "lateral")
        _require_gsd_range(self.gsd_along_min_m, self.gsd_along_max_m, "along")
        if not math.isfinite(self.gsd_reference_m) or self.gsd_reference_m <= 0.0:
            raise ValueError("gsd_reference_m must be finite and > 0")
        if len(self.dataset_hash) != 64:
            raise ValueError("dataset_hash must be 64 hex characters")
        return self


def compute_dataset_hash(dataset_dir: str | Path) -> str:
    """Return a SHA-256 over every file except ``dataset.json``.

    Args:
        dataset_dir: Finished dataset directory.

    Returns:
        str: Lowercase hex digest.

    Raises:
        FileNotFoundError: If the directory has no hashed files.

    Notes:
        The digest input is ``relative_posix_path:file_sha256`` lines in
        sorted path order.
    """
    root = Path(dataset_dir)
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != _MANIFEST_NAME
    )
    if not files:
        raise FileNotFoundError(f"no dataset files under {root}")
    hasher = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix()
        file_hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                file_hasher.update(chunk)
        hasher.update(f"{rel}:{file_hasher.hexdigest()}\n".encode())
    return hasher.hexdigest()


def write_manifest(path: str | Path, manifest: DatasetManifest) -> None:
    """Write ``dataset.json``.

    Args:
        path: Destination file.
        manifest: Identity to store.

    Returns:
        None.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(_to_payload(manifest), indent=2) + "\n", encoding="utf-8")


def load_manifest(
    path: str | Path,
    *,
    verify: bool = True,
) -> DatasetManifest:
    """Parse ``dataset.json``.

    Args:
        path: JSON path.
        verify: When True, recompute the content hash and compare it.

    Returns:
        DatasetManifest: Loaded identity.

    Raises:
        OSError / json.JSONDecodeError: On a missing or malformed file.
        ValueError: If the payload does not match the schema or the hash differs.
        FileNotFoundError: If verification is on and the directory has no shard files.
    """
    dest = Path(path)
    raw = json.loads(dest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{dest.name} must be a JSON object")
    payload = dict(raw)
    if "schema" not in payload:
        raise ValueError("dataset.json missing schema")
    payload["schema_version"] = payload.pop("schema")
    if payload["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"dataset schema {payload['schema_version']} is unsupported; "
            f"rebuild the dataset (current schema {SCHEMA_VERSION})"
        )
    manifest = TypeAdapter(DatasetManifest).validate_python(payload)
    if verify:
        digest = compute_dataset_hash(dest.parent)
        if digest != manifest.dataset_hash:
            raise ValueError(
                f"dataset hash mismatch: dataset.json has {manifest.dataset_hash}, "
                f"files have {digest}"
            )
    return manifest


def _require_gsd_range(low: float, high: float, axis: str) -> None:
    """Raise ValueError when a GSD range is not finite and ordered.

    Args:
        low: Minimum metres.
        high: Maximum metres.
        axis: ``lateral`` or ``along``, used in the message.

    Returns:
        None.

    Raises:
        ValueError: If either end is not finite and positive, or ``low > high``.
    """
    if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high <= 0.0:
        raise ValueError(f"{axis} gsd range must be finite and > 0; got {low}..{high}")
    if low > high:
        raise ValueError(f"{axis} gsd min {low} exceeds max {high}")


def _to_payload(manifest: DatasetManifest) -> dict[str, object]:
    """Convert a manifest into a JSON-ready mapping.

    Args:
        manifest: Identity.

    Returns:
        dict[str, object]: Keys in file order. ``schema_version`` is written
        as ``schema``.
    """
    return {
        "schema": manifest.schema_version,
        "source": manifest.source,
        "source_ref": manifest.source_ref,
        "weight_table_id": manifest.weight_table_id,
        "band_names": list(manifest.band_names),
        "norm": manifest.norm,
        "image_dtype": manifest.image_dtype,
        "gsd_reference_m": manifest.gsd_reference_m,
        "split": {
            "seed": manifest.split.seed,
            "train_fraction": manifest.split.train_fraction,
            "val_fraction": manifest.split.val_fraction,
            "test_fraction": manifest.split.test_fraction,
        },
        "augment": {"elements": list(manifest.augment.elements)},
        "bins": [_bin_payload(item) for item in manifest.bins],
        "shards": [_shard_payload(item) for item in manifest.shards],
        "gsd_lateral_min_m": manifest.gsd_lateral_min_m,
        "gsd_lateral_max_m": manifest.gsd_lateral_max_m,
        "gsd_along_min_m": manifest.gsd_along_min_m,
        "gsd_along_max_m": manifest.gsd_along_max_m,
        "dataset_hash": manifest.dataset_hash,
    }


def _bin_payload(item: BinRecord) -> dict[str, object]:
    """Convert one bin record to JSON.

    Args:
        item: Bin row.

    Returns:
        dict[str, object]: JSON object.
    """
    return {
        "bin_id": item.bin_id,
        "lateral_m": item.lateral_m,
        "along_m": item.along_m,
        "elevation_deg": item.elevation_deg,
    }


def _shard_payload(item: ShardCount) -> dict[str, object]:
    """Convert one shard count to JSON.

    Args:
        item: Shard row.

    Returns:
        dict[str, object]: JSON object.
    """
    return {
        "task": item.task,
        "split": item.split,
        "height": item.height,
        "width": item.width,
        "n": item.n,
        "n_positive": item.n_positive,
    }


def shard_dir(dataset_dir: Path, task: str, split: str, height: int, width: int) -> Path:
    """Return the directory of one shard.

    Args:
        dataset_dir: Finished dataset root.
        task: ``classifier`` or ``segmentor``.
        split: ``train``, ``val``, or ``test``.
        height: Tile H.
        width: Tile W.

    Returns:
        Path: ``<dataset>/<task>/<split>/<H>x<W>``.
    """
    return dataset_dir / task / split / f"{height}x{width}"


def parse_shard_size(name: str) -> tuple[int, int] | None:
    """Parse an ``<H>x<W>`` directory name.

    Args:
        name: Directory name.

    Returns:
        tuple[int, int] | None: Height and width, or None when the name does
        not match.
    """
    if "x" not in name:
        return None
    left, right = name.split("x", maxsplit=1)
    if not left.isdecimal() or not right.isdecimal():
        return None
    return (int(left), int(right))
