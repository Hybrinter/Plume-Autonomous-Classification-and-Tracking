"""Frozen training-figure recipe serialization into bundle artifacts.

The codec turns caller-supplied ``TrainingFigure`` recipes into
checksummed ``BundleFile`` bytes: a canonical
``training-figure-data.json`` document carrying the full recipe records
(markers, scales, run status, reasons, and warnings included), a flat
typed point table in CSV and Parquet, and the table-schema manifest. It
never re-derives history coordinates, reads run histories, models, or
datasets, or renders anything; every value is copied from the frozen
records and table codecs run in a private temporary directory. Repeated
``n_exposures`` values are per-series metadata, not additive support.
No summary is published here.

Contains:
  - TRAINING_POINTS_SCHEMA, TABLE_SCHEMAS: exact column contracts for
    the emitted tables.
  - TrainingArtifacts: returned file bytes and references.
  - training_artifacts: the ``Result`` codec boundary.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.artifacts import (
    BundleFile,
    ColumnSpec,
    Scalar,
    TableSchema,
    artifact_ref,
    write_table,
)
from tools.ml_models.analysis.contracts import ArtifactRef, check_bundle_path
from tools.ml_models.analysis.training_figures import TrainingFigure

_DATA_PATH = "training-figure-data.json"
_SCHEMAS_PATH = "tables/training_schema.json"

TRAINING_POINTS_SCHEMA = TableSchema(
    columns=(
        ColumnSpec(name="figure", dtype="STRING"),
        ColumnSpec(name="series", dtype="STRING"),
        ColumnSpec(name="point_index", dtype="INT64"),
        ColumnSpec(name="x", dtype="FLOAT64"),
        ColumnSpec(name="y", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="n_exposures", dtype="INT64"),
        ColumnSpec(name="exposure_unit", dtype="STRING"),
        ColumnSpec(name="population", dtype="STRING"),
        ColumnSpec(name="points_only", dtype="BOOLEAN"),
    )
)

TABLE_SCHEMAS: dict[str, TableSchema] = {
    "tables/training_points.csv": TRAINING_POINTS_SCHEMA,
    "tables/training_points.parquet": TRAINING_POINTS_SCHEMA,
}


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    """Encoded training-figure bundle members and their checksum references."""

    files: tuple[BundleFile, ...]
    references: tuple[ArtifactRef, ...]


def _canonical(data: object) -> bytes:
    """Serialize one frozen structure deterministically; NaN fails closed."""
    return json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _point_rows(figures: tuple[TrainingFigure, ...]) -> tuple[dict[str, Scalar], ...]:
    """Flatten one verbatim row per captured point; zeros, nulls and duplicate x stay."""
    rows: list[dict[str, Scalar]] = []
    for record in figures:
        for series in record.series:
            for index, (x, y) in enumerate(zip(series.x, series.y, strict=True)):
                rows.append(
                    {
                        "figure": record.identifier,
                        "series": series.name,
                        "point_index": index,
                        "x": x,
                        "y": y,
                        "n_exposures": series.n_exposures,
                        "exposure_unit": series.exposure_unit,
                        "population": series.population,
                        "points_only": series.points_only,
                    }
                )
    return tuple(rows)


def training_artifacts(
    figures: tuple[TrainingFigure, ...],
    *,
    prefix: str = "",
) -> Result[TrainingArtifacts, str]:
    """Encode frozen training-figure recipes into bundle files and references.

    ``training-figure-data.json`` is the canonical serialization of the
    supplied recipes under ``{"schema_version": 1, "figures": [...]}``.
    ``tables/training_points.{csv,parquet}`` carry one verbatim row per
    series point against ``TRAINING_POINTS_SCHEMA`` and
    ``tables/training_schema.json`` declares the contract. Table codecs
    stage in a private temporary directory; failures return ``Err``
    before any caller output is touched. A nonempty ``prefix``
    namespaces every returned file and reference path under
    ``prefix/...`` without changing bytes, checksums, or the relative
    paths inside the documents.
    """
    if prefix:
        try:
            check_bundle_path(prefix + "/payload.json")
        except ValueError as exc:
            return Err(f"unsafe artifact prefix: {exc}")
    try:
        data_bytes = _canonical(
            {"schema_version": 1, "figures": [asdict(record) for record in figures]}
        )
        schemas_bytes = _canonical(
            {rel: asdict(TABLE_SCHEMAS[rel]) for rel in sorted(TABLE_SCHEMAS)}
        )
        point_rows = _point_rows(figures)
        files: list[BundleFile] = []
        refs: list[ArtifactRef] = []
        data_ref = artifact_ref(
            _DATA_PATH, data_bytes, kind="REFERENCE", format="json", rows=len(figures)
        )
        if isinstance(data_ref, Err):
            return data_ref
        refs.append(data_ref.value)
        files.append(BundleFile(path=_DATA_PATH, data=data_bytes))
        with tempfile.TemporaryDirectory(prefix=".training-figures-") as staging:
            staged = Path(staging)
            for rel in sorted(TABLE_SCHEMAS):
                table_file = staged.joinpath(*rel.split("/"))
                table_file.parent.mkdir(parents=True, exist_ok=True)
                table_ref = write_table(table_file, point_rows, TABLE_SCHEMAS[rel], bundle_path=rel)
                if isinstance(table_ref, Err):
                    return table_ref
                refs.append(table_ref.value)
                files.append(BundleFile(path=rel, data=table_file.read_bytes()))
        schemas_ref = artifact_ref(_SCHEMAS_PATH, schemas_bytes, kind="REFERENCE", format="json")
        if isinstance(schemas_ref, Err):
            return schemas_ref
        refs.append(schemas_ref.value)
        files.append(BundleFile(path=_SCHEMAS_PATH, data=schemas_bytes))
    except (OSError, TypeError, ValueError, OverflowError, RuntimeError) as exc:
        return Err(f"training figure artifact serialization failed: {exc}")
    if prefix:
        files = [replace(file, path=prefix + "/" + file.path) for file in files]
        refs = [replace(ref, path=prefix + "/" + ref.path) for ref in refs]
    return Ok(TrainingArtifacts(tuple(files), tuple(refs)))
