"""Model evidence bundle assembly: frozen documents, tables and capture bytes.

Assembly is mechanical only: existing artifact codecs serialize the frozen
training/task/generalization recipes and evidence, capture bytes are carried
verbatim, and one combined prediction manifest covers every evaluated split.
``model-evidence.json`` is the canonical frozen scientific document that a
render-only publisher re-verifies against the summary without touching
models, checkpoints or source datasets. ``tables/split_metrics`` persist
every recorded cohort and stratum ``MetricValue`` verbatim; compact scalar
rows stay in the captured ``rows.jsonl`` files.

Contains:
  - SPLIT_METRICS_SCHEMA, SPLIT_METRICS_TABLES: split metric row contract.
  - ModelArtifacts: assembled files plus content-addressed references.
  - model_evidence_document: the canonical ``model-evidence.json`` payload.
  - assemble_model_artifacts: the ``Result`` assembly boundary.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.artifacts import (
    BundleFile,
    ColumnSpec,
    Scalar,
    TableSchema,
    artifact_file_ref,
    artifact_ref,
    write_table,
)
from tools.ml_models.analysis.config import ModelAnalysisConfig, write_config
from tools.ml_models.analysis.contracts import (
    ArtifactRef,
    MetricValue,
    SplitEvidence,
)
from tools.ml_models.analysis.generalization_artifacts import generalization_artifacts
from tools.ml_models.analysis.model_figure_artifacts import model_figure_artifacts
from tools.ml_models.analysis.model_measurement import ModelMeasurement
from tools.ml_models.analysis.prediction_artifacts import prediction_artifacts
from tools.ml_models.analysis.prediction_selections import PredictionGallery
from tools.ml_models.analysis.training_artifacts import training_artifacts
from tools.ml_models.analysis.visuals.predictions import (
    PredictionPreview,
    PredictionPreviewCapture,
)

_EVIDENCE_PATH = "model-evidence.json"
_CONFIG_PATH = "config.toml"
_SPLIT_METRICS_SCHEMA_PATH = "tables/split_metrics_schema.json"

SPLIT_METRICS_SCHEMA = TableSchema(
    columns=(
        ColumnSpec(name="split", dtype="STRING"),
        ColumnSpec(name="task", dtype="STRING"),
        ColumnSpec(name="population", dtype="STRING"),
        ColumnSpec(name="stratum_name", dtype="STRING", nullable=True),
        ColumnSpec(name="stratum_value", dtype="STRING", nullable=True),
        ColumnSpec(name="metric", dtype="STRING"),
        ColumnSpec(name="value", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="status", dtype="STRING"),
        ColumnSpec(name="reason", dtype="STRING", nullable=True),
        ColumnSpec(name="unit", dtype="STRING"),
        ColumnSpec(name="aggregation", dtype="STRING"),
        ColumnSpec(name="threshold", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="support_unit", dtype="STRING"),
        ColumnSpec(name="support_n", dtype="INT64"),
        ColumnSpec(name="support_counts_json", dtype="STRING"),
        ColumnSpec(name="interval_lower", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="interval_upper", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="interval_confidence", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="interval_method", dtype="STRING", nullable=True),
        ColumnSpec(name="interval_replicates", dtype="INT64", nullable=True),
        ColumnSpec(name="interval_valid", dtype="INT64", nullable=True),
        ColumnSpec(name="interval_seed", dtype="INT64", nullable=True),
    )
)

SPLIT_METRICS_TABLES: dict[str, TableSchema] = {
    "tables/split_metrics.parquet": SPLIT_METRICS_SCHEMA,
    "tables/split_metrics.csv": SPLIT_METRICS_SCHEMA,
}


def _canonical(values: object) -> bytes:
    """Encode finite canonical JSON without creating an output directory."""
    return json.dumps(values, sort_keys=True, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _metric_row(
    evidence: SplitEvidence,
    metric: MetricValue,
    population: str,
    stratum_name: str | None,
    stratum_value: str | None,
) -> dict[str, Scalar]:
    """Flatten one frozen MetricValue; no measurement or aggregation happens here."""
    interval = metric.interval
    return {
        "split": evidence.split,
        "task": evidence.task,
        "population": population,
        "stratum_name": stratum_name,
        "stratum_value": stratum_value,
        "metric": metric.name,
        "value": metric.value,
        "status": metric.status,
        "reason": metric.reason,
        "unit": metric.unit,
        "aggregation": metric.aggregation,
        "threshold": metric.threshold,
        "support_unit": metric.support.unit,
        "support_n": metric.support.n,
        "support_counts_json": json.dumps(
            [asdict(count) for count in metric.support.counts],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "interval_lower": interval.lower if interval is not None else None,
        "interval_upper": interval.upper if interval is not None else None,
        "interval_confidence": interval.confidence if interval is not None else None,
        "interval_method": interval.method if interval is not None else None,
        "interval_replicates": interval.n_replicates if interval is not None else None,
        "interval_valid": interval.n_valid if interval is not None else None,
        "interval_seed": interval.seed if interval is not None else None,
    }


def _split_metric_rows(splits: tuple[SplitEvidence, ...]) -> tuple[dict[str, Scalar], ...]:
    """Flatten every recorded cohort and stratum metric of the measured splits."""
    rows: list[dict[str, Scalar]] = []
    for evidence in splits:
        rows.extend(
            _metric_row(evidence, metric, "cohort", None, None) for metric in evidence.metrics
        )
        for stratum in evidence.strata:
            rows.extend(
                _metric_row(evidence, metric, "stratum", stratum.name, stratum.value)
                for metric in stratum.metrics
            )
    return tuple(rows)


def model_evidence_document(measured: ModelMeasurement, cfg: ModelAnalysisConfig) -> bytes:
    """Serialize the canonical frozen scientific document, version 1.

    The document carries identities, the resolved analysis configuration and
    every frozen scientific record (baseline, generalization, development,
    resources, metric reference, availability, warnings); it contains no
    model weights, tensors or rendered artifacts, and no duplicate summary.
    """
    return _canonical(
        {
            "schema_version": 1,
            "kind": "MODEL_ANALYSIS_EVIDENCE",
            "training_dataset": asdict(measured.training_dataset),
            "evaluation_dataset": asdict(measured.evaluation_dataset),
            "checkpoint": asdict(measured.checkpoint),
            "config_digest": measured.config_digest,
            "config": asdict(cfg),
            "baseline": asdict(measured.baseline),
            "generalization": [asdict(record) for record in measured.generalization],
            "development": (
                asdict(measured.development) if measured.development is not None else None
            ),
            "resources": [asdict(record) for record in measured.resources],
            "metric_reference": [asdict(record) for record in measured.reference],
            "outputs": [asdict(record) for record in measured.outputs],
            "warnings": list(measured.warnings),
            "source_snapshots": [
                {
                    "path": file.path,
                    "sha256": hashlib.sha256(file.data).hexdigest(),
                    "size_bytes": len(file.data),
                }
                for file in measured.snapshots
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class ModelArtifacts:
    """Bundle bytes and references for everything except rendered outputs."""

    files: tuple[BundleFile, ...]
    refs: tuple[ArtifactRef, ...]


def combined_prediction_capture(measured: ModelMeasurement) -> PredictionPreviewCapture:
    """Union the frozen per-split preview captures into one manifest input."""
    previews: list[PredictionPreview] = []
    files: list[BundleFile] = []
    galleries: list[PredictionGallery] = []
    for capture in measured.captures:
        previews.extend(capture.previews.previews)
        files.extend(capture.previews.files)
        galleries.extend(capture.previews.galleries)
    return PredictionPreviewCapture(tuple(previews), tuple(files), tuple(galleries))


def _deduplicate(files: list[BundleFile], refs: list[ArtifactRef]) -> Result[ModelArtifacts, str]:
    """Keep exactly one entry per path; identical bytes and refs only."""
    file_map: dict[str, bytes] = {}
    ref_map: dict[str, ArtifactRef] = {}
    for file in files:
        previous_bytes = file_map.get(file.path)
        if previous_bytes is None:
            file_map[file.path] = file.data
        elif previous_bytes != file.data:
            return Err(f"conflicting bundle bytes for {file.path}")
    for ref in refs:
        previous_ref = ref_map.get(ref.path)
        if previous_ref is None:
            ref_map[ref.path] = ref
        elif previous_ref != ref:
            return Err(f"conflicting bundle references for {ref.path}")
    missing = [path for path in ref_map if path not in file_map]
    if missing:
        return Err(f"bundle references lack files: {sorted(missing)}")
    unreferenced = [path for path in file_map if path not in ref_map]
    if unreferenced:
        return Err(f"bundle files lack references: {sorted(unreferenced)}")
    return Ok(
        ModelArtifacts(
            tuple(BundleFile(path, file_map[path]) for path in sorted(file_map)),
            tuple(ref_map[path] for path in sorted(ref_map)),
        )
    )


def assemble_model_artifacts(
    measured: ModelMeasurement, cfg: ModelAnalysisConfig
) -> Result[ModelArtifacts, str]:
    """Freeze one model measurement into bundle files and references.

    Chart documents and typed tables come from the existing codecs under
    ``task/<kind>/`` and ``generalization/<split>/`` prefixes. One shared
    ``prediction-manifest.json`` covers every evaluated split's normalized
    capture paths. Capture bytes, source snapshots and the frozen
    ``model-evidence.json`` document are carried verbatim. Nothing is
    written to the caller's output path.
    """
    try:
        kind = measured.checkpoint.kind
        files: list[BundleFile] = []
        refs: list[ArtifactRef] = []
        training = training_artifacts(measured.training_figures)
        if isinstance(training, Err):
            return training
        files.extend(training.value.files)
        refs.extend(training.value.references)
        task = model_figure_artifacts(measured.task_figures, prefix=f"task/{kind}")
        if isinstance(task, Err):
            return task
        files.extend(task.value.files)
        refs.extend(task.value.references)
        strata_figures = model_figure_artifacts(
            measured.generalization_figures, prefix="generalization"
        )
        if isinstance(strata_figures, Err):
            return strata_figures
        files.extend(strata_figures.value.files)
        refs.extend(strata_figures.value.references)
        development_bound = measured.development is not None and any(
            record.split == "val" for record in measured.generalization
        )
        for record in measured.generalization:
            development = (
                measured.development if development_bound and record.split == "val" else None
            )
            batch = generalization_artifacts(
                record, development=development, prefix=f"generalization/{record.split}"
            )
            if isinstance(batch, Err):
                return batch
            files.extend(batch.value.files)
            refs.extend(batch.value.refs)
        predictions = prediction_artifacts(combined_prediction_capture(measured))
        if isinstance(predictions, Err):
            return predictions
        files.extend(predictions.value.files)
        refs.extend(predictions.value.references)
        captures = measured.captures + (
            (measured.baseline_capture,) if measured.baseline_capture is not None else ()
        )
        for capture in captures:
            files.extend(capture.files)
            refs.extend(capture.evidence.artifacts)
        files.extend(measured.snapshots)
        for file in measured.snapshots:
            snapshot_ref = artifact_ref(
                file.path,
                file.data,
                kind="REFERENCE",
                format=file.path.rsplit(".", 1)[-1],
                population="source_snapshot",
            )
            if isinstance(snapshot_ref, Err):
                return snapshot_ref
            refs.append(snapshot_ref.value)
        evidence_bytes = model_evidence_document(measured, cfg)
        evidence_ref = artifact_ref(_EVIDENCE_PATH, evidence_bytes, kind="REFERENCE", format="json")
        if isinstance(evidence_ref, Err):
            return evidence_ref
        files.append(BundleFile(_EVIDENCE_PATH, evidence_bytes))
        refs.append(evidence_ref.value)
        with tempfile.TemporaryDirectory(prefix=".model-evidence-") as staging:
            staged = Path(staging)
            config_file = staged / _CONFIG_PATH
            written = write_config(config_file, cfg)
            if isinstance(written, Err):
                return written
            config_ref = artifact_file_ref(
                config_file, bundle_path=_CONFIG_PATH, kind="CONFIG", format="toml"
            )
            if isinstance(config_ref, Err):
                return config_ref
            files.append(BundleFile(_CONFIG_PATH, config_file.read_bytes()))
            refs.append(config_ref.value)
            metric_rows = _split_metric_rows(
                tuple(capture.evidence for capture in measured.captures)
            )
            for rel in sorted(SPLIT_METRICS_TABLES):
                table_file = staged.joinpath(*rel.split("/"))
                table_file.parent.mkdir(parents=True, exist_ok=True)
                table_ref = write_table(
                    table_file, metric_rows, SPLIT_METRICS_TABLES[rel], bundle_path=rel
                )
                if isinstance(table_ref, Err):
                    return table_ref
                files.append(BundleFile(rel, table_file.read_bytes()))
                refs.append(table_ref.value)
            schema_bytes = _canonical(
                {rel: asdict(SPLIT_METRICS_TABLES[rel]) for rel in SPLIT_METRICS_TABLES}
            )
            schema_ref = artifact_ref(
                _SPLIT_METRICS_SCHEMA_PATH, schema_bytes, kind="REFERENCE", format="json"
            )
            if isinstance(schema_ref, Err):
                return schema_ref
            files.append(BundleFile(_SPLIT_METRICS_SCHEMA_PATH, schema_bytes))
            refs.append(schema_ref.value)
        return _deduplicate(files, refs)
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
        return Err(f"model artifact assembly failed before publication: {exc}")
