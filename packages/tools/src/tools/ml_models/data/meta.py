"""Dataset identity, provenance, and the processed-pack hash.

Contains:
  - NormName, IngestPath, Radiometry: closed strings stored in the sidecars.
  - DatasetMeta: every field written to ``dataset.json``.
  - Provenance: every field written to ``provenance.json``.
  - provenance_from_meta / dataset_meta_from_provenance.
  - compute_dataset_hash / write_dataset_meta / load_dataset_meta.
  - write_provenance / load_provenance.

The hash covers ``images.npy``, ``masks.npy``, ``labels.npy``, ``splits.json``,
and ``provenance.json``. ``dataset.json`` is not an input. ``band_z`` provenance
stores the fitted per-band mean and population std. Other recipes store empty
moment lists.

Numeric fields use strict validation. A JSON boolean or numeric string is
rejected. A JSON integer is accepted for a float field. JSON arrays for
``band_names``, ``band_mean``, and ``band_std`` become tuples. Dataclass-level
``strict=True`` is not set: on a pydantic dataclass it rejects a JSON object.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

NormName = Literal["normalize_dn", "band_z", "unit"]
IngestPath = Literal[
    "sentinel2_4250706_prism_proxy",
    "flight_camera",
    "sentinel2_4250706_study",
]
Radiometry = Literal["s2_l2a_reflectance", "normalize_dn"]

_SCHEMA = ConfigDict(extra="forbid")
_JsonInt = Annotated[int, Field(strict=True)]
_JsonFloat = Annotated[float, Field(strict=True)]
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_PACK_HASH_FILES: tuple[str, ...] = (
    "images.npy",
    "masks.npy",
    "labels.npy",
    "splits.json",
    "provenance.json",
)
_PROVENANCE_FIELDS: tuple[str, ...] = (
    "ingest_path",
    "radiometry",
    "gsd_m",
    "extent_m",
    "weight_table_id",
    "band_names",
    "norm",
    "bit_depth",
    "band_mean",
    "band_std",
)
_DATASET_META_FIELDS: tuple[str, ...] = (
    "dataset_hash",
    "source_doi",
    "n",
    "height",
    "width",
    "in_channels",
    "band_names",
    "norm",
    "bit_depth",
    "ingest_path",
    "radiometry",
    "gsd_m",
    "extent_m",
    "weight_table_id",
    "band_mean",
    "band_std",
)


@pydantic_dataclass(frozen=True, slots=True, config=_SCHEMA)
class Provenance:
    """Sidecar stored in ``provenance.json`` and copied onto ``DatasetMeta``.

    Attributes:
        ingest_path: How the pack was built.
        radiometry: Radiometric meaning of the stored planes.
        gsd_m: Ground sample distance in meters.
        extent_m: Tile extent in meters.
        weight_table_id: Class-weight table identifier.
        band_names: Channel names, length ``C``.
        norm: Normalization recipe applied before the pack was written.
        bit_depth: ADC bit depth used by ``normalize_dn``.
        band_mean: Fitted per-band means. Empty unless ``norm`` is ``band_z``.
        band_std: Fitted per-band population standard deviations. Empty unless
            ``norm`` is ``band_z``. Each value is greater than 0.
    """

    ingest_path: IngestPath
    radiometry: Radiometry
    gsd_m: _JsonFloat
    extent_m: _JsonFloat
    weight_table_id: str
    band_names: tuple[str, ...]
    norm: NormName
    bit_depth: _JsonInt
    band_mean: tuple[_JsonFloat, ...] = ()
    band_std: tuple[_JsonFloat, ...] = ()

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        """Reject an empty band list, a bit depth below 1, or bad band moments."""
        if len(self.band_names) < 1:
            raise ValueError("band_names must be non-empty")
        if self.bit_depth < 1:
            raise ValueError(f"bit_depth must be >= 1; got {self.bit_depth}")
        _require_band_moments(self.norm, len(self.band_names), self.band_mean, self.band_std)
        return self


@pydantic_dataclass(frozen=True, slots=True, config=_SCHEMA)
class DatasetMeta:
    """Identity sidecar for a processed pack.

    Attributes:
        dataset_hash: Lowercase SHA-256 over the pack file digests.
        source_doi: Dataset DOI string. Empty when synthetic.
        n: Sample count N.
        height: Spatial height H.
        width: Spatial width W.
        in_channels: Band count C.
        band_names: Channel names. Length equals ``in_channels``.
        norm: Normalization recipe name.
        bit_depth: ADC bit depth.
        ingest_path: How the pack was built.
        radiometry: Radiometric meaning of the stored planes.
        gsd_m: Ground sample distance in meters.
        extent_m: Tile extent in meters.
        weight_table_id: Class-weight table identifier.
        band_mean: Fitted per-band means. Empty unless ``norm`` is ``band_z``.
        band_std: Fitted per-band population standard deviations. Empty unless
            ``norm`` is ``band_z``. Each value is greater than 0.
    """

    dataset_hash: str
    source_doi: str
    n: _JsonInt
    height: _JsonInt
    width: _JsonInt
    in_channels: _JsonInt
    band_names: tuple[str, ...]
    norm: NormName
    bit_depth: _JsonInt
    ingest_path: IngestPath
    radiometry: Radiometry
    gsd_m: _JsonFloat
    extent_m: _JsonFloat
    weight_table_id: str
    band_mean: tuple[_JsonFloat, ...] = ()
    band_std: tuple[_JsonFloat, ...] = ()

    @model_validator(mode="after")
    def _bounds(self) -> Self:
        """Reject non-positive geometry, a band-count mismatch, or bad band moments."""
        if self.n < 1:
            raise ValueError(f"n must be >= 1; got {self.n}")
        if self.height < 1 or self.width < 1:
            raise ValueError(f"height and width must be >= 1; got {self.height}x{self.width}")
        if self.in_channels < 1:
            raise ValueError(f"in_channels must be >= 1; got {self.in_channels}")
        if self.in_channels != len(self.band_names):
            raise ValueError(
                f"in_channels {self.in_channels} != len(band_names) {len(self.band_names)}"
            )
        if self.bit_depth < 1:
            raise ValueError(f"bit_depth must be >= 1; got {self.bit_depth}")
        _require_band_moments(self.norm, len(self.band_names), self.band_mean, self.band_std)
        return self


def provenance_from_meta(meta: DatasetMeta) -> Provenance:
    """Copy the provenance fields out of a dataset sidecar.

    Args:
        meta: Pack identity.

    Returns:
        Provenance: Ingest, radiometry, geometry, bands, norm, bit depth, and
        band moments.
    """
    return Provenance(
        ingest_path=meta.ingest_path,
        radiometry=meta.radiometry,
        gsd_m=meta.gsd_m,
        extent_m=meta.extent_m,
        weight_table_id=meta.weight_table_id,
        band_names=meta.band_names,
        norm=meta.norm,
        bit_depth=meta.bit_depth,
        band_mean=meta.band_mean,
        band_std=meta.band_std,
    )


def dataset_meta_from_provenance(
    provenance: Provenance,
    *,
    dataset_hash: str,
    source_doi: str,
    n: int,
    height: int,
    width: int,
) -> DatasetMeta:
    """Build a dataset sidecar from provenance plus pack geometry.

    Args:
        provenance: Fields stored in ``provenance.json``.
        dataset_hash: Lowercase pack digest. Empty when the pack is only in memory.
        source_doi: Dataset DOI string.
        n: Sample count N.
        height: Spatial height H.
        width: Spatial width W.

    Returns:
        DatasetMeta: Identity whose ``in_channels`` is ``len(provenance.band_names)``.

    Raises:
        ValueError: If geometry or bit depth is out of range.
    """
    return DatasetMeta(
        dataset_hash=dataset_hash,
        source_doi=source_doi,
        n=n,
        height=height,
        width=width,
        in_channels=len(provenance.band_names),
        band_names=provenance.band_names,
        norm=provenance.norm,
        bit_depth=provenance.bit_depth,
        ingest_path=provenance.ingest_path,
        radiometry=provenance.radiometry,
        gsd_m=provenance.gsd_m,
        extent_m=provenance.extent_m,
        weight_table_id=provenance.weight_table_id,
        band_mean=provenance.band_mean,
        band_std=provenance.band_std,
    )


def compute_dataset_hash(pack_dir: str | Path) -> str:
    """Return a SHA-256 over the pack file digests.

    Args:
        pack_dir: Directory with ``images.npy``, ``masks.npy``, ``labels.npy``,
            ``splits.json``, and ``provenance.json``.

    Returns:
        str: Lowercase hex digest.

    Raises:
        FileNotFoundError: If a required pack file is missing.

    Notes:
        The hash is the SHA-256 of ``name:file_sha256`` lines, one per file, in
        the order listed above. Each file is digested in 8 MiB chunks. The
        digest does not include ``dataset.json``.
    """
    root = Path(pack_dir)
    hasher = hashlib.sha256()
    for name in _PACK_HASH_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing pack file {path}")
        file_hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                file_hasher.update(chunk)
        hasher.update(f"{name}:{file_hasher.hexdigest()}\n".encode())
    return hasher.hexdigest()


def write_dataset_meta(path: str | Path, meta: DatasetMeta) -> None:
    """Write every DatasetMeta field as JSON.

    Args:
        path: Destination ``dataset.json``.
        meta: Pack identity.

    Returns:
        None.
    """
    payload: dict[str, object] = {
        "dataset_hash": meta.dataset_hash,
        "source_doi": meta.source_doi,
        "n": meta.n,
        "height": meta.height,
        "width": meta.width,
        "in_channels": meta.in_channels,
        "band_names": list(meta.band_names),
        "norm": meta.norm,
        "bit_depth": meta.bit_depth,
        "ingest_path": meta.ingest_path,
        "radiometry": meta.radiometry,
        "gsd_m": meta.gsd_m,
        "extent_m": meta.extent_m,
        "weight_table_id": meta.weight_table_id,
        "band_mean": list(meta.band_mean),
        "band_std": list(meta.band_std),
    }
    _write_json(Path(path), payload)


def load_dataset_meta(
    path: str | Path,
    *,
    pack_dir: Path | None = None,
    verify: bool = True,
) -> DatasetMeta:
    """Parse ``dataset.json`` into DatasetMeta.

    Args:
        path: JSON path.
        pack_dir: Directory of the hashed pack files. Verification runs only
            when this is set.
        verify: When True and ``pack_dir`` is set, recompute the pack hash.

    Returns:
        DatasetMeta: Loaded identity.

    Raises:
        OSError / json.JSONDecodeError: On a missing or malformed file.
        ValueError: If keys do not match the sidecar, a value has the wrong
            type, or the recomputed hash does not match ``dataset_hash``.
        FileNotFoundError: If verification is on and a hashed pack file is missing.
    """
    dest = Path(path)
    data = _read_json_object(dest)
    _require_exact_keys(data, _DATASET_META_FIELDS, dest.name)
    _require_string_list(data, "band_names")
    _require_number_list(data, "band_mean")
    _require_number_list(data, "band_std")
    meta = TypeAdapter(DatasetMeta).validate_python(data)
    if verify and pack_dir is not None:
        digest = compute_dataset_hash(pack_dir)
        if digest != meta.dataset_hash:
            raise ValueError(
                f"dataset hash mismatch: dataset.json has {meta.dataset_hash}, pack has {digest}"
            )
    return meta


def write_provenance(path: str | Path, provenance: Provenance) -> None:
    """Write every Provenance field as JSON.

    Args:
        path: Destination ``provenance.json``.
        provenance: Ingest and normalization record.

    Returns:
        None.
    """
    payload: dict[str, object] = {
        "ingest_path": provenance.ingest_path,
        "radiometry": provenance.radiometry,
        "gsd_m": provenance.gsd_m,
        "extent_m": provenance.extent_m,
        "weight_table_id": provenance.weight_table_id,
        "band_names": list(provenance.band_names),
        "norm": provenance.norm,
        "bit_depth": provenance.bit_depth,
        "band_mean": list(provenance.band_mean),
        "band_std": list(provenance.band_std),
    }
    _write_json(Path(path), payload)


def load_provenance(path: str | Path) -> Provenance:
    """Parse ``provenance.json`` into Provenance.

    Args:
        path: JSON path.

    Returns:
        Provenance: Loaded ingest and normalization record.

    Raises:
        OSError / json.JSONDecodeError: On a missing or malformed file.
        ValueError: If keys do not match the sidecar or a value has the wrong type.
    """
    dest = Path(path)
    data = _read_json_object(dest)
    _require_exact_keys(data, _PROVENANCE_FIELDS, dest.name)
    _require_string_list(data, "band_names")
    _require_number_list(data, "band_mean")
    _require_number_list(data, "band_std")
    return TypeAdapter(Provenance).validate_python(data)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write a JSON object with a trailing newline.

    Args:
        path: Destination file. Parent directories are created.
        payload: JSON object. ``band_names`` is a list.

    Returns:
        None.

    Raises:
        ValueError: If a float is NaN or infinity.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, object]:
    """Parse a JSON object.

    Args:
        path: File path.

    Returns:
        dict[str, object]: Decoded mapping.

    Raises:
        OSError / json.JSONDecodeError: On a missing or malformed file.
        ValueError: If the root is not a JSON object or a key is not a string.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    decoded: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"{path.name} keys must be strings")
        decoded[key] = value
    return decoded


def _require_exact_keys(data: Mapping[str, object], fields: tuple[str, ...], label: str) -> None:
    """Raise when ``data`` does not have exactly ``fields``.

    Args:
        data: Decoded JSON object.
        fields: Required keys.
        label: File name used in the error.

    Returns:
        None.

    Raises:
        ValueError: If a key is missing or unknown.
    """
    expected = set(fields)
    found = set(data)
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch; missing={missing} extra={extra}")


def _require_band_moments(
    norm: NormName,
    band_count: int,
    band_mean: tuple[float, ...],
    band_std: tuple[float, ...],
) -> None:
    """Reject band moments that do not match the normalization recipe.

    Args:
        norm: Normalization recipe name.
        band_count: Number of bands.
        band_mean: Per-band means. Empty unless ``norm`` is ``band_z``.
        band_std: Per-band population standard deviations. Empty unless ``norm``
            is ``band_z``.

    Returns:
        None.

    Raises:
        ValueError: If ``band_z`` moments do not have length ``band_count``, a
            std is not positive, or another recipe stores moments.
    """
    if norm == "band_z":
        if len(band_mean) != band_count or len(band_std) != band_count:
            raise ValueError(
                f"band_z band_mean length {len(band_mean)} and band_std length {len(band_std)} "
                f"must both equal band count {band_count}"
            )
        if any(value <= 0.0 for value in band_std):
            raise ValueError("band_std values must be > 0")
        return
    if len(band_mean) != 0 or len(band_std) != 0:
        raise ValueError(f"norm {norm!r} records empty band_mean and band_std")


def _require_string_list(data: Mapping[str, object], key: str) -> None:
    """Raise when ``data[key]`` is not a list of strings.

    Args:
        data: Decoded JSON object. ``key`` is present.
        key: Field name.

    Returns:
        None.

    Raises:
        ValueError: If the value is not a list of strings.
    """
    value = data[key]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")


def _require_number_list(data: Mapping[str, object], key: str) -> None:
    """Raise when ``data[key]`` is not a list of numbers.

    Args:
        data: Decoded JSON object. ``key`` is present.
        key: Field name.

    Returns:
        None.

    Raises:
        ValueError: If the value is not a list of ints or floats. Booleans are
            rejected.
    """
    value = data[key]
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"{key} must be a list of numbers")
