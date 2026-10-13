"""Frozen generalization evidence serialization into bundle artifacts.

The codec turns an already-measured ``GeneralizationEvidence`` (and an
optional ``DevelopmentEvidence``) into checksummed ``BundleFile`` bytes:
canonical ``generalization.json``/``development.json``/``metric-reference.json``
documents, a flat typed metric table, an interval audit table, and the
table-schema manifest. It never measures, resamples, reloads sources, or
reads models/config; every value is copied from the frozen records, table
codecs run in a private temporary directory, and unknown metric
definitions stay explicit unknowns.

Contains:
  - GENERALIZATION_METRICS_SCHEMA, INTERVAL_AUDIT_SCHEMA, TABLE_SCHEMAS:
    exact column contracts for the emitted tables.
  - GeneralizationArtifacts: returned file bytes and references.
  - generalization_artifacts: the ``Result`` codec boundary.

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
from tools.ml_models.analysis.contracts import (
    ArtifactRef,
    MetricValue,
    check_bundle_path,
)
from tools.ml_models.analysis.metrics.generalization import (
    DevelopmentEvidence,
    GeneralizationEvidence,
    metric_reference,
)

_GENERALIZATION_PATH = "generalization.json"
_DEVELOPMENT_PATH = "development.json"
_REFERENCE_PATH = "metric-reference.json"
_SCHEMAS_PATH = "tables/schema.json"

GENERALIZATION_METRICS_SCHEMA = TableSchema(
    columns=(
        ColumnSpec(name="population", dtype="STRING"),
        ColumnSpec(name="stratum_name", dtype="STRING", nullable=True),
        ColumnSpec(name="stratum_value", dtype="STRING", nullable=True),
        ColumnSpec(name="name", dtype="STRING"),
        ColumnSpec(name="value", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="status", dtype="STRING"),
        ColumnSpec(name="reason", dtype="STRING", nullable=True),
        ColumnSpec(name="unit", dtype="STRING"),
        ColumnSpec(name="aggregation", dtype="STRING"),
        ColumnSpec(name="support_unit", dtype="STRING"),
        ColumnSpec(name="support_n", dtype="INT64"),
        ColumnSpec(name="threshold", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="interval_lower", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="interval_upper", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="interval_confidence", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="interval_method", dtype="STRING", nullable=True),
        ColumnSpec(name="interval_replicates", dtype="INT64", nullable=True),
        ColumnSpec(name="interval_valid", dtype="INT64", nullable=True),
        ColumnSpec(name="interval_seed", dtype="INT64", nullable=True),
    )
)

INTERVAL_AUDIT_SCHEMA = TableSchema(
    columns=(
        ColumnSpec(name="population", dtype="STRING"),
        ColumnSpec(name="stratum_name", dtype="STRING", nullable=True),
        ColumnSpec(name="stratum_value", dtype="STRING", nullable=True),
        ColumnSpec(name="metric", dtype="STRING"),
        ColumnSpec(name="n_replicates", dtype="INT64"),
        ColumnSpec(name="n_attempted", dtype="INT64"),
        ColumnSpec(name="n_valid", dtype="INT64"),
        ColumnSpec(name="n_invalid", dtype="INT64"),
        ColumnSpec(name="n_groups", dtype="INT64"),
        ColumnSpec(name="reason", dtype="STRING", nullable=True),
    )
)

TABLE_SCHEMAS: dict[str, TableSchema] = {
    "tables/generalization_metrics.parquet": GENERALIZATION_METRICS_SCHEMA,
    "tables/interval_audit.csv": INTERVAL_AUDIT_SCHEMA,
}


@dataclass(frozen=True, slots=True)
class GeneralizationArtifacts:
    """Encoded generalization bundle members and their checksum references."""

    files: tuple[BundleFile, ...]
    refs: tuple[ArtifactRef, ...]


def _canonical(data: object) -> bytes:
    """Serialize one frozen structure deterministically; NaN fails closed."""
    return json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _metric_row(
    metric: MetricValue,
    population: str,
    stratum_name: str | None,
    stratum_value: str | None,
) -> dict[str, Scalar]:
    """Flatten one frozen metric; interval fields stay null when absent."""
    interval = metric.interval
    return {
        "population": population,
        "stratum_name": stratum_name,
        "stratum_value": stratum_value,
        "name": metric.name,
        "value": metric.value,
        "status": metric.status,
        "reason": metric.reason,
        "unit": metric.unit,
        "aggregation": metric.aggregation,
        "support_unit": metric.support.unit,
        "support_n": metric.support.n,
        "threshold": metric.threshold,
        "interval_lower": interval.lower if interval is not None else None,
        "interval_upper": interval.upper if interval is not None else None,
        "interval_confidence": interval.confidence if interval is not None else None,
        "interval_method": interval.method if interval is not None else None,
        "interval_replicates": interval.n_replicates if interval is not None else None,
        "interval_valid": interval.n_valid if interval is not None else None,
        "interval_seed": interval.seed if interval is not None else None,
    }


def _metric_rows(evidence: GeneralizationEvidence) -> tuple[dict[str, Scalar], ...]:
    """Flatten cohort, stratum, and baseline metrics in evidence order."""
    rows = [_metric_row(metric, "cohort", None, None) for metric in evidence.metrics]
    for stratum in evidence.strata:
        rows += [
            _metric_row(metric, "stratum", stratum.name, stratum.value)
            for metric in stratum.metrics
        ]
    rows += [_metric_row(metric, "baseline", None, None) for metric in evidence.baseline_metrics]
    return tuple(rows)


def _audit_rows(evidence: GeneralizationEvidence) -> tuple[dict[str, Scalar], ...]:
    """Flatten the verbatim replicate audit.

    ``n_attempted`` is zero only when bootstrap resampling never ran for
    the metric (fewer than two recorded groups); metrics whose
    replicates ran but produced no valid value keep their attempted and
    invalid counts.
    """
    rows: list[dict[str, Scalar]] = [
        {
            "population": "cohort",
            "stratum_name": None,
            "stratum_value": None,
            "metric": audit.metric,
            "n_replicates": audit.n_replicates,
            "n_attempted": audit.n_attempted,
            "n_valid": audit.n_valid,
            "n_invalid": audit.n_invalid,
            "n_groups": audit.n_groups,
            "reason": audit.reason,
        }
        for audit in evidence.intervals
    ]
    for stratum in evidence.stratum_intervals:
        rows += [
            {
                "population": "stratum",
                "stratum_name": stratum.name,
                "stratum_value": stratum.value,
                "metric": audit.metric,
                "n_replicates": audit.n_replicates,
                "n_attempted": audit.n_attempted,
                "n_valid": audit.n_valid,
                "n_invalid": audit.n_invalid,
                "n_groups": audit.n_groups,
                "reason": audit.reason,
            }
            for audit in stratum.intervals
        ]
    return tuple(rows)


def generalization_artifacts(
    evidence: GeneralizationEvidence,
    *,
    development: DevelopmentEvidence | None = None,
    prefix: str = "",
) -> Result[GeneralizationArtifacts, str]:
    """Encode frozen generalization evidence into bundle files and references.

    ``generalization.json`` is the canonical serialization of the supplied
    evidence, including the frozen baseline spec and curves.
    ``development.json`` is emitted only when a caller-bound development
    record is supplied, and only when that record shares the evidence's
    dataset, manifest, scoring, and task identity.
    ``metric-reference.json``
    lists definition lookups for every metric name in the evidence plus
    development gap names; unknown names keep their explicit lookup error.
    Table codecs stage in a private temporary directory; failures return
    ``Err`` before any caller output is touched. A nonempty ``prefix``
    namespaces every returned file and reference path under
    ``prefix/...`` without changing bytes, checksums, or the relative
    paths inside the documents.
    """
    if prefix:
        try:
            check_bundle_path(prefix + "/payload.json")
        except ValueError as exc:
            return Err(f"unsafe artifact prefix: {exc}")
    if development is not None and (
        development.dataset_hash != evidence.dataset_hash
        or development.dataset_manifest_hash != evidence.dataset_manifest_hash
        or development.score_config != evidence.score_config
        or development.task != evidence.task
    ):
        return Err(
            "development evidence does not share the captured dataset, "
            "manifest, scoring, or task identity"
        )
    names = [
        metric.name
        for metric in (
            *evidence.metrics,
            *evidence.baseline_metrics,
            *(metric for stratum in evidence.strata for metric in stratum.metrics),
        )
    ]
    if development is not None:
        names += [gap.name for gap in development.gaps]
    try:
        with tempfile.TemporaryDirectory(prefix=".generalization-") as staging:
            staged = Path(staging)
            documents: list[tuple[str, bytes]] = [
                (_GENERALIZATION_PATH, _canonical(asdict(evidence))),
                (_REFERENCE_PATH, _canonical([asdict(e) for e in metric_reference(tuple(names))])),
            ]
            if development is not None:
                documents.append((_DEVELOPMENT_PATH, _canonical(asdict(development))))
            table_rows = {
                "tables/generalization_metrics.parquet": _metric_rows(evidence),
                "tables/interval_audit.csv": _audit_rows(evidence),
            }
            files: list[BundleFile] = []
            refs: list[ArtifactRef] = []
            for path, data in documents:
                ref = artifact_ref(path, data, kind="REFERENCE", format="json")
                if isinstance(ref, Err):
                    return ref
                refs.append(ref.value)
                files.append(BundleFile(path=path, data=data))
            for rel in sorted(TABLE_SCHEMAS):
                table_file = staged.joinpath(*rel.split("/"))
                table_file.parent.mkdir(parents=True, exist_ok=True)
                table_ref = write_table(
                    table_file, table_rows[rel], TABLE_SCHEMAS[rel], bundle_path=rel
                )
                if isinstance(table_ref, Err):
                    return table_ref
                refs.append(table_ref.value)
                files.append(BundleFile(path=rel, data=table_file.read_bytes()))
            schemas_bytes = _canonical(
                {rel: asdict(TABLE_SCHEMAS[rel]) for rel in sorted(TABLE_SCHEMAS)}
            )
            schemas_ref = artifact_ref(
                _SCHEMAS_PATH, schemas_bytes, kind="REFERENCE", format="json"
            )
            if isinstance(schemas_ref, Err):
                return schemas_ref
            refs.append(schemas_ref.value)
            files.append(BundleFile(path=_SCHEMAS_PATH, data=schemas_bytes))
    except (OSError, TypeError, ValueError, OverflowError, RuntimeError) as exc:
        return Err(f"generalization artifact serialization failed: {exc}")
    if prefix:
        files = [replace(file, path=prefix + "/" + file.path) for file in files]
        refs = [replace(ref, path=prefix + "/" + ref.path) for ref in refs]
    return Ok(GeneralizationArtifacts(tuple(files), tuple(refs)))
