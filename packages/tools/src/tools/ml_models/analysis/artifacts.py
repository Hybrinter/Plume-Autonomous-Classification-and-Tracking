"""Versioned evidence codecs, identities, publication, and typed tables.

Summaries serialize as canonical sorted-key UTF-8 JSON; bundle files are
content-addressed. ``publish_bundle`` reserves output directories exclusively,
writes a ``.incomplete`` marker for in-progress or non-complete bundles, and
leaves the marker in place whenever verification should refuse the result.
``verify_bundle`` recomputes every referenced checksum inside the resolved
bundle root. CSV tables use a canonical scalar encoding: ``\\N`` is the only
null marker, a leading backslash escapes strings that start with one, integers
are decimal, floats use ``repr``, and booleans are ``true``/``false``.

Contains:
  - encode_summary / decode_summary: canonical tagged summary JSON.
  - checksum_file, dataset_identity, artifact_ref, artifact_file_ref.
  - BundleFile, publish_bundle, verify_bundle.
  - ColumnSpec, TableSchema, write_table, read_table.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, field
from pathlib import Path, PurePosixPath
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from flight.libs.types import Err, Ok, Result
from pydantic import ConfigDict, StrictBool, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass

from tools.ml_models.analysis.contracts import (
    ArtifactKind,
    ArtifactRef,
    DatasetIdentity,
    check_bundle_path,
)
from tools.ml_models.analysis.summaries import (
    DatasetSummary,
    ModelTrainingSummary,
    Summary,
)
from tools.ml_models.dataset.manifest import load_manifest

SUMMARY_FILENAME = "summary.json"
INCOMPLETE_FILENAME = ".incomplete"
_MANIFEST_NAME = "dataset.json"
_CHUNK_BYTES = 8 * 1024 * 1024
_SCHEMA = ConfigDict(extra="forbid")

type Scalar = str | int | float | bool | None
type ColumnDtype = Literal["STRING", "INT64", "FLOAT64", "BOOLEAN"]
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_NULL = "\\N"
_INT_RE = re.compile(r"^-?\d+$")

_DATASET_ADAPTER = TypeAdapter(DatasetSummary)
_MODEL_ADAPTER = TypeAdapter(ModelTrainingSummary)


def encode_summary(summary: Summary) -> Result[bytes, str]:
    """Serialize a summary to canonical sorted-key UTF-8 JSON.

    Tuples encode as JSON arrays; ``NaN`` and ``Infinity`` are refused.
    """
    try:
        payload = asdict(summary)
        text = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        return Err(f"cannot encode summary: {exc}")
    return Ok(text.encode("utf-8"))


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-JSON constant {value}")


def decode_summary(raw: bytes) -> Result[Summary, str]:
    """Parse canonical summary JSON, dispatching on the ``summary_kind`` tag."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return Err(f"summary is not UTF-8: {exc}")
    try:
        payload = json.loads(text, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        return Err(f"summary is not valid JSON: {exc}")
    if not isinstance(payload, dict):
        return Err("summary JSON must be an object")
    kind = payload.get("summary_kind")
    try:
        if kind == "DATASET_ANALYSIS":
            return Ok(_DATASET_ADAPTER.validate_python(payload))
        if kind == "MODEL_TRAINING_ANALYSIS":
            return Ok(_MODEL_ADAPTER.validate_python(payload))
    except ValueError as exc:
        return Err(f"invalid summary: {exc}")
    return Err(f"unknown summary_kind {kind!r}")


def checksum_file(path: Path) -> Result[str, str]:
    """Stream a SHA-256 over the file at ``path``."""
    hasher = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while True:
                chunk = handle.read(_CHUNK_BYTES)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError as exc:
        return Err(f"cannot checksum {path}: {exc}")
    return Ok(hasher.hexdigest())


def dataset_identity(root: Path) -> Result[DatasetIdentity, str]:
    """Verify a finished dataset and return its identity record.

    ``manifest_hash`` covers the actual ``dataset.json`` bytes; the content
    hash comes from the verified manifest and excludes that file. The dataset
    is never written or mutated.
    """
    manifest_path = Path(root) / _MANIFEST_NAME
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        return Err(f"cannot read manifest {manifest_path}: {exc}")
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        return Err(f"invalid dataset manifest {manifest_path}: {exc}")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        identity = DatasetIdentity(
            content_hash=manifest.dataset_hash,
            manifest_hash=manifest_hash,
            schema_version=manifest.schema_version,
            source=manifest.source,
            band_names=manifest.band_names,
            gsd_reference_m=manifest.gsd_reference_m,
        )
    except ValueError as exc:
        return Err(f"manifest fields cannot form an identity: {exc}")
    return Ok(identity)


def artifact_ref(
    path: str,
    data: bytes,
    *,
    kind: ArtifactKind,
    format: str,
    rows: int | None = None,
    population: str | None = None,
) -> Result[ArtifactRef, str]:
    """Build an ArtifactRef whose hash and size derive from ``data``."""
    try:
        ref = ArtifactRef(
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            kind=kind,
            format=format,
            rows=rows,
            population=population,
        )
    except ValueError as exc:
        return Err(f"invalid artifact reference: {exc}")
    return Ok(ref)


def artifact_file_ref(
    file: Path,
    *,
    bundle_path: str,
    kind: ArtifactKind,
    format: str,
    rows: int | None = None,
    population: str | None = None,
) -> Result[ArtifactRef, str]:
    """Build an ArtifactRef from a real file's bytes under a bundle path."""
    checksum = checksum_file(file)
    if isinstance(checksum, Err):
        return Err(checksum.error)
    try:
        size = Path(file).stat().st_size
    except OSError as exc:
        return Err(f"cannot stat {file}: {exc}")
    try:
        ref = ArtifactRef(
            path=bundle_path,
            sha256=checksum.value,
            size_bytes=size,
            kind=kind,
            format=format,
            rows=rows,
            population=population,
        )
    except ValueError as exc:
        return Err(f"invalid artifact reference: {exc}")
    return Ok(ref)


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class BundleFile:
    """One file slated for exclusive publication into a bundle.

    Attributes:
        path: Safe relative POSIX path inside the bundle.
        data: File bytes; checksums derive from this content.
    """

    path: str
    data: bytes

    @model_validator(mode="after")
    def _bounds(self) -> BundleFile:
        check_bundle_path(self.path)
        if self.path in (SUMMARY_FILENAME, INCOMPLETE_FILENAME):
            raise ValueError(f"bundle path {self.path!r} is reserved")
        return self


def _summary_refs(summary: Summary) -> Result[tuple[ArtifactRef, ...], str]:
    candidates = list(summary.artifacts)
    for split in summary.splits:
        candidates.extend(split.artifacts)
    if isinstance(summary, ModelTrainingSummary) and summary.history is not None:
        candidates.append(summary.history)
    refs: dict[str, ArtifactRef] = {}
    for ref in candidates:
        existing = refs.get(ref.path)
        if existing is None:
            refs[ref.path] = ref
        elif existing != ref:
            return Err(f"conflicting references for bundle path {ref.path!r}")
    return Ok(tuple(refs.values()))


def _matches_refs(
    refs: tuple[ArtifactRef, ...], files: tuple[BundleFile, ...]
) -> Result[dict[str, bytes], str]:
    ref_by_path = {ref.path: ref for ref in refs}
    file_by_path: dict[str, bytes] = {}
    for item in files:
        if item.path in file_by_path:
            return Err(f"duplicate bundle file path {item.path!r}")
        file_by_path[item.path] = item.data
    if set(file_by_path) != set(ref_by_path):
        missing = sorted(set(ref_by_path) - set(file_by_path))
        extra = sorted(set(file_by_path) - set(ref_by_path))
        return Err(f"bundle files do not match references; missing {missing}, unreferenced {extra}")
    for ref in refs:
        data = file_by_path[ref.path]
        if len(data) != ref.size_bytes:
            return Err(f"size mismatch for {ref.path}")
        if hashlib.sha256(data).hexdigest() != ref.sha256:
            return Err(f"checksum mismatch for {ref.path}")
    return Ok(file_by_path)


def _contained(path: Path, root: Path) -> bool:
    resolved_root = root.resolve()
    resolved = path.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def publish_bundle(
    out: Path,
    summary: Summary,
    files: tuple[BundleFile, ...],
    dataset_root: Path | None = None,
) -> Result[Path, str]:
    """Publish one evidence bundle into a fresh, exclusively-reserved output.

    References must match the supplied bytes exactly, and only ``COMPLETE``
    summaries publish; other statuses fail before the output is reserved. A
    ``.incomplete`` marker is written first and removed only after every
    file and the summary land; failed writes keep the marker so
    ``verify_bundle`` refuses the bundle. Existing directories are never
    overwritten or deleted.
    """
    out = Path(out)
    refs = _summary_refs(summary)
    if isinstance(refs, Err):
        return Err(refs.error)
    matched = _matches_refs(refs.value, files)
    if isinstance(matched, Err):
        return Err(matched.error)
    encoded = encode_summary(summary)
    if isinstance(encoded, Err):
        return Err(encoded.error)
    if summary.status != "COMPLETE":
        return Err(f"only COMPLETE summaries publish; status is {summary.status}")
    if dataset_root is not None:
        source = Path(dataset_root).resolve()
        if _contained(out, source):
            return Err(f"output {out} lies inside the source dataset {dataset_root}")
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return Err(f"output {out} already exists; refusing to overwrite")
    except OSError as exc:
        return Err(f"cannot reserve {out}: {exc}")
    marker = out / INCOMPLETE_FILENAME
    try:
        marker.touch(exist_ok=False)
        for path in sorted(matched.value):
            target = out.joinpath(*PurePosixPath(path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(matched.value[path])
        with (out / SUMMARY_FILENAME).open("xb") as handle:
            handle.write(encoded.value)
        marker.unlink()
    except OSError as exc:
        return Err(f"bundle write failed; marker retained at {out}: {exc}")
    return Ok(out)


def _target_is_safe(root: Path, rel: str) -> bool:
    current = root
    for part in PurePosixPath(rel).parts:
        current = current / part
        if os.path.islink(current) or os.path.isjunction(current):
            return False
    return _contained(current, root)


def verify_bundle(root: Path) -> Result[Summary, str]:
    """Verify a complete published bundle and return its summary.

    The incomplete marker, a non-complete status, an escaped or linked
    target, and any missing or corrupt referenced file all fail closed. The
    summary file itself is never part of its own checksum list.
    """
    root = Path(root)
    if not root.is_dir():
        return Err(f"bundle {root} is not a directory")
    if (root / INCOMPLETE_FILENAME).exists():
        return Err(f"bundle {root} is marked incomplete")
    summary_path = root / SUMMARY_FILENAME
    if not summary_path.is_file() or summary_path.is_symlink():
        return Err(f"bundle {root} has no summary")
    try:
        raw = summary_path.read_bytes()
    except OSError as exc:
        return Err(f"cannot read summary: {exc}")
    decoded = decode_summary(raw)
    if isinstance(decoded, Err):
        return Err(decoded.error)
    summary = decoded.value
    if summary.status != "COMPLETE":
        return Err(f"bundle {root} has status {summary.status}")
    refs = _summary_refs(summary)
    if isinstance(refs, Err):
        return Err(refs.error)
    for ref in refs.value:
        if not _target_is_safe(root, ref.path):
            return Err(f"reference {ref.path} escapes the bundle root")
        target = root.joinpath(*PurePosixPath(ref.path).parts)
        if not target.is_file():
            return Err(f"reference {ref.path} is missing")
        checksum = checksum_file(target)
        if isinstance(checksum, Err):
            return Err(checksum.error)
        if checksum.value != ref.sha256:
            return Err(f"reference {ref.path} checksum mismatch")
        try:
            size = target.stat().st_size
        except OSError as exc:
            return Err(f"reference {ref.path} cannot be stat'ed: {exc}")
        if size != ref.size_bytes:
            return Err(f"reference {ref.path} size mismatch")
    return Ok(summary)


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class ColumnSpec:
    """One typed, named column.

    Attributes:
        name: Column name, unique within its schema.
        dtype: ``STRING``, ``INT64``, ``FLOAT64``, or ``BOOLEAN``.
        nullable: Whether the column accepts None.
    """

    name: str
    dtype: ColumnDtype
    nullable: StrictBool = field(default=False)

    @model_validator(mode="after")
    def _bounds(self) -> ColumnSpec:
        if not self.name:
            raise ValueError("column name must be nonempty")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class TableSchema:
    """Ordered, uniquely-named typed columns for a flat table.

    Attributes:
        columns: Column specifications in file order.
    """

    columns: tuple[ColumnSpec, ...]

    @model_validator(mode="after")
    def _bounds(self) -> TableSchema:
        if not self.columns:
            raise ValueError("a table schema requires at least one column")
        names = [column.name for column in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("column names must be unique")
        return self


def _check_scalar(value: Scalar, column: ColumnSpec) -> Result[None, str]:
    if value is None:
        if column.nullable:
            return Ok(None)
        return Err(f"column {column.name} is not nullable")
    if column.dtype == "STRING":
        if isinstance(value, str):
            return Ok(None)
        return Err(f"column {column.name} requires a string")
    if column.dtype == "INT64":
        if isinstance(value, int) and not isinstance(value, bool):
            if _INT64_MIN <= value <= _INT64_MAX:
                return Ok(None)
            return Err(f"column {column.name} value is outside int64 range")
        return Err(f"column {column.name} requires an exact int64")
    if column.dtype == "FLOAT64":
        if isinstance(value, bool):
            return Err(f"column {column.name} requires a finite float64")
        if isinstance(value, (int, float)):
            try:
                converted = float(value)
            except OverflowError, ValueError:
                return Err(f"column {column.name} value does not fit a float64")
            if math.isfinite(converted):
                return Ok(None)
        return Err(f"column {column.name} requires a finite float64")
    if column.dtype == "BOOLEAN":
        if isinstance(value, bool):
            return Ok(None)
        return Err(f"column {column.name} requires a boolean")
    return Err(f"unknown dtype {column.dtype}")


def _encode_cell(value: Scalar, column: ColumnSpec) -> Result[str, str]:
    checked = _check_scalar(value, column)
    if isinstance(checked, Err):
        return Err(checked.error)
    if value is None:
        return Ok(_NULL)
    if column.dtype == "STRING":
        assert isinstance(value, str)
        return Ok("\\" + value if value.startswith("\\") else value)
    if column.dtype == "INT64":
        return Ok(str(value))
    if column.dtype == "FLOAT64":
        return Ok(repr(float(value)))
    assert isinstance(value, bool)
    return Ok("true" if value else "false")


def _decode_cell(cell: str, column: ColumnSpec) -> Result[Scalar, str]:
    if cell == _NULL:
        if column.nullable:
            return Ok(None)
        return Err(f"column {column.name} is not nullable")
    if column.dtype == "STRING":
        if cell.startswith("\\\\"):
            return Ok(cell[1:])
        if cell.startswith("\\"):
            return Err(f"column {column.name} cell {cell!r} has a malformed escape")
        return Ok(cell)
    if cell == "":
        return Err(f"column {column.name} has an empty non-string cell")
    if column.dtype == "INT64":
        if _INT_RE.fullmatch(cell) is None:
            return Err(f"column {column.name} cell {cell!r} is not an int64")
        value = int(cell)
        if not _INT64_MIN <= value <= _INT64_MAX:
            return Err(f"column {column.name} cell {cell!r} exceeds int64")
        return Ok(value)
    if column.dtype == "FLOAT64":
        try:
            number = float(cell)
        except ValueError:
            return Err(f"column {column.name} cell {cell!r} is not a float")
        if not math.isfinite(number):
            return Err(f"column {column.name} cell {cell!r} is not finite")
        return Ok(number)
    if cell == "true":
        return Ok(True)
    if cell == "false":
        return Ok(False)
    return Err(f"column {column.name} cell {cell!r} is not a boolean")


def _check_row_keys(row: dict[str, Scalar], schema: TableSchema) -> Result[None, str]:
    expected = {column.name for column in schema.columns}
    if set(row) != expected:
        missing = sorted(expected - set(row))
        extra = sorted(set(row) - expected)
        return Err(f"row keys do not match the schema; missing {missing}, extra {extra}")
    return Ok(None)


_ARROW_TYPES = {
    "STRING": pa.string(),
    "INT64": pa.int64(),
    "FLOAT64": pa.float64(),
    "BOOLEAN": pa.bool_(),
}


def _write_csv(
    path: Path, rows: tuple[dict[str, Scalar], ...], schema: TableSchema
) -> Result[None, str]:
    for row in rows:
        keys = _check_row_keys(row, schema)
        if isinstance(keys, Err):
            return Err(keys.error)
    lines: list[list[str]] = [[column.name for column in schema.columns]]
    for row in rows:
        encoded: list[str] = []
        for column in schema.columns:
            cell = _encode_cell(row[column.name], column)
            if isinstance(cell, Err):
                return Err(cell.error)
            encoded.append(cell.value)
        lines.append(encoded)
    try:
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerows(lines)
    except FileExistsError:
        return Err(f"table {path} already exists; refusing to overwrite")
    except OSError as exc:
        return Err(f"cannot write table {path}: {exc}")
    return Ok(None)


def _arrow_cell(value: Scalar, column: ColumnSpec) -> Scalar:
    if column.dtype == "FLOAT64" and value is not None:
        return float(value)
    return value


def _write_parquet(
    path: Path, rows: tuple[dict[str, Scalar], ...], schema: TableSchema
) -> Result[None, str]:
    for row in rows:
        keys = _check_row_keys(row, schema)
        if isinstance(keys, Err):
            return Err(keys.error)
        for column in schema.columns:
            checked = _check_scalar(row[column.name], column)
            if isinstance(checked, Err):
                return Err(checked.error)
    fields = [
        pa.field(column.name, _ARROW_TYPES[column.dtype], nullable=column.nullable)
        for column in schema.columns
    ]
    arrow_schema = pa.schema(fields)
    try:
        columns = {
            column.name: pa.array(
                [_arrow_cell(row[column.name], column) for row in rows],
                type=_ARROW_TYPES[column.dtype],
            )
            for column in schema.columns
        }
        table = pa.table(columns, schema=arrow_schema)
    except (pa.ArrowException, OverflowError, ValueError) as exc:
        return Err(f"cannot build parquet table: {exc}")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        return Err(f"table {path} already exists; refusing to overwrite")
    except OSError as exc:
        return Err(f"cannot write table {path}: {exc}")
    try:
        with os.fdopen(fd, "wb") as handle:
            pq.write_table(table, handle)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowException) as exc:
        return Err(f"cannot write parquet table {path}: {exc}")
    return Ok(None)


def write_table(
    path: Path,
    rows: tuple[dict[str, Scalar], ...],
    schema: TableSchema,
    *,
    bundle_path: str | None = None,
) -> Result[ArtifactRef, str]:
    """Write ``rows`` as typed CSV or Parquet, selected by suffix.

    Every row must carry exactly the schema's keys; scalars are checked
    against the column dtypes and nullability. The file is created
    exclusively. The returned reference names ``bundle_path`` (or the file
    name) with a checksum and size derived from the written bytes.
    """
    path = Path(path)
    rel = bundle_path if bundle_path is not None else path.name
    try:
        check_bundle_path(rel)
    except ValueError as exc:
        return Err(str(exc))
    if path.suffix == ".csv":
        written = _write_csv(path, rows, schema)
    elif path.suffix == ".parquet":
        written = _write_parquet(path, rows, schema)
    else:
        return Err(f"unsupported table suffix {path.suffix!r}; use .csv or .parquet")
    if isinstance(written, Err):
        return Err(written.error)
    fmt = path.suffix.removeprefix(".")
    return artifact_file_ref(path, bundle_path=rel, kind="TABLE", format=fmt, rows=len(rows))


def _read_csv(path: Path, schema: TableSchema) -> Result[tuple[dict[str, Scalar], ...], str]:
    names = [column.name for column in schema.columns]
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            records = list(csv.reader(handle, strict=True))
    except (OSError, csv.Error, UnicodeError) as exc:
        return Err(f"cannot read table {path}: {exc}")
    if not records or records[0] != names:
        return Err(f"table {path} header does not match the schema")
    rows: list[dict[str, Scalar]] = []
    for record in records[1:]:
        if len(record) != len(schema.columns):
            return Err(f"table {path} row has {len(record)} cells, expected {len(schema.columns)}")
        decoded_row: dict[str, Scalar] = {}
        for cell, column in zip(record, schema.columns, strict=True):
            decoded = _decode_cell(cell, column)
            if isinstance(decoded, Err):
                return Err(decoded.error)
            decoded_row[column.name] = decoded.value
        rows.append(decoded_row)
    return Ok(tuple(rows))


def _read_parquet(path: Path, schema: TableSchema) -> Result[tuple[dict[str, Scalar], ...], str]:
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowException) as exc:
        return Err(f"cannot read parquet table {path}: {exc}")
    arrow_schema = table.schema
    if arrow_schema.names != [column.name for column in schema.columns]:
        return Err(f"parquet table {path} columns do not match the schema")
    for field_, column in zip(arrow_schema, schema.columns, strict=True):
        if field_.type != _ARROW_TYPES[column.dtype]:
            return Err(
                f"parquet column {column.name} has type {field_.type}, "
                f"expected {_ARROW_TYPES[column.dtype]}"
            )
        if field_.nullable != column.nullable:
            return Err(f"parquet column {column.name} nullability mismatch")
    rows: list[dict[str, Scalar]] = []
    for record in table.to_pylist():
        checked = _check_row_keys(record, schema)
        if isinstance(checked, Err):
            return Err(checked.error)
        for column in schema.columns:
            scalar = _check_scalar(record[column.name], column)
            if isinstance(scalar, Err):
                return Err(scalar.error)
        rows.append({column.name: record[column.name] for column in schema.columns})
    return Ok(tuple(rows))


def read_table(path: Path, schema: TableSchema) -> Result[tuple[dict[str, Scalar], ...], str]:
    """Read a typed CSV or Parquet table back under ``schema``."""
    path = Path(path)
    if path.suffix == ".csv":
        return _read_csv(path, schema)
    if path.suffix == ".parquet":
        return _read_parquet(path, schema)
    return Err(f"unsupported table suffix {path.suffix!r}; use .csv or .parquet")
