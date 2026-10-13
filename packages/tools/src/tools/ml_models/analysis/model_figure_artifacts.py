"""Frozen model-figure recipe serialization into bundle artifacts.

The codec turns caller-supplied ``ModelFigure`` recipes into checksummed
``BundleFile`` bytes: a canonical ``model-figure-data.json`` document
carrying the full recipe records (identity, population, operating
points, matrices, ranges, reasons and notes included), a flat typed
point table in CSV and Parquet, and the table-schema manifest. Matrices
and notes live only in the JSON document; the point table copies one
verbatim row per series point, zeros, nulls and duplicate x included.
Repeated ``support_n`` values are per-series metadata, not additive
support. No values are re-derived, no sources, histories, models, or
datasets are read, and no summary is published here.

Contains:
  - MODEL_POINTS_SCHEMA, TABLE_SCHEMAS: exact column contracts.
  - ModelFigureArtifacts: returned file bytes and references.
  - model_figure_artifacts: the ``Result`` codec boundary.

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
from tools.ml_models.analysis.model_figures import ModelFigure

_DATA_PATH = "model-figure-data.json"
_SCHEMAS_PATH = "tables/model_schema.json"

MODEL_POINTS_SCHEMA = TableSchema(
    columns=(
        ColumnSpec(name="figure", dtype="STRING"),
        ColumnSpec(name="task", dtype="STRING"),
        ColumnSpec(name="split", dtype="STRING"),
        ColumnSpec(name="dataset_hash", dtype="STRING"),
        ColumnSpec(name="dataset_manifest_hash", dtype="STRING", nullable=True),
        ColumnSpec(name="checkpoint_hash", dtype="STRING", nullable=True),
        ColumnSpec(name="series", dtype="STRING"),
        ColumnSpec(name="point_index", dtype="INT64"),
        ColumnSpec(name="x", dtype="FLOAT64"),
        ColumnSpec(name="y", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="lower", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="upper", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="support_unit", dtype="STRING"),
        ColumnSpec(name="support_n", dtype="INT64"),
        ColumnSpec(name="point_support_n", dtype="INT64", nullable=True),
        ColumnSpec(name="style", dtype="STRING"),
    )
)

TABLE_SCHEMAS: dict[str, TableSchema] = {
    "tables/model_points.csv": MODEL_POINTS_SCHEMA,
    "tables/model_points.parquet": MODEL_POINTS_SCHEMA,
}


@dataclass(frozen=True, slots=True)
class ModelFigureArtifacts:
    """Encoded model-figure bundle members and their checksum references."""

    files: tuple[BundleFile, ...]
    references: tuple[ArtifactRef, ...]


def _canonical(data: object) -> bytes:
    """Serialize one frozen structure deterministically; NaN fails closed."""
    return json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _point_rows(
    figures: tuple[ModelFigure, ...],
) -> Result[tuple[dict[str, Scalar], ...], str]:
    """Flatten one verbatim row per captured series point; matrices stay in JSON."""
    rows: list[dict[str, Scalar]] = []
    for record in figures:
        for series in record.series:
            if len(series.x) != len(series.y):
                return Err(f"series {series.name} coordinates have mismatched lengths")
            for edge in (series.lower, series.upper, series.point_support):
                if edge and len(edge) != len(series.x):
                    return Err(f"series {series.name} has misaligned interval or support")
            for index, (x, y) in enumerate(zip(series.x, series.y, strict=True)):
                rows.append(
                    {
                        "figure": record.identifier,
                        "task": record.identity.task,
                        "split": record.identity.split,
                        "dataset_hash": record.identity.dataset_hash,
                        "dataset_manifest_hash": record.identity.dataset_manifest_hash,
                        "checkpoint_hash": record.identity.checkpoint_hash,
                        "series": series.name,
                        "point_index": index,
                        "x": x,
                        "y": y,
                        "lower": series.lower[index] if series.lower else None,
                        "upper": series.upper[index] if series.upper else None,
                        "support_unit": series.support.unit,
                        "support_n": series.support.n,
                        "point_support_n": (
                            series.point_support[index] if series.point_support else None
                        ),
                        "style": series.style,
                    }
                )
    return Ok(tuple(rows))


def model_figure_artifacts(
    figures: tuple[ModelFigure, ...],
    *,
    prefix: str = "",
) -> Result[ModelFigureArtifacts, str]:
    """Encode frozen model-figure recipes into bundle files and references.

    ``model-figure-data.json`` is the canonical serialization of the
    supplied recipes under ``{"schema_version": 1, "figures": [...]}``.
    ``tables/model_points.{csv,parquet}`` carry one verbatim row per
    series point against ``MODEL_POINTS_SCHEMA`` and
    ``tables/model_schema.json`` declares the contract. Table codecs
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
        if isinstance(point_rows, Err):
            return point_rows
        files: list[BundleFile] = []
        refs: list[ArtifactRef] = []
        data_ref = artifact_ref(
            _DATA_PATH, data_bytes, kind="REFERENCE", format="json", rows=len(figures)
        )
        if isinstance(data_ref, Err):
            return data_ref
        refs.append(data_ref.value)
        files.append(BundleFile(path=_DATA_PATH, data=data_bytes))
        with tempfile.TemporaryDirectory(prefix=".model-figures-") as staging:
            staged = Path(staging)
            for rel in sorted(TABLE_SCHEMAS):
                table_file = staged.joinpath(*rel.split("/"))
                table_file.parent.mkdir(parents=True, exist_ok=True)
                table_ref = write_table(
                    table_file, point_rows.value, TABLE_SCHEMAS[rel], bundle_path=rel
                )
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
        return Err(f"model figure artifact serialization failed: {exc}")
    if prefix:
        files = [replace(file, path=prefix + "/" + file.path) for file in files]
        refs = [replace(ref, path=prefix + "/" + ref.path) for ref in refs]
    return Ok(ModelFigureArtifacts(tuple(files), tuple(refs)))
