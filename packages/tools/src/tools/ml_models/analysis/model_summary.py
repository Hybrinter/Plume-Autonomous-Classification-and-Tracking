"""One model/training summary bound to frozen scientific evidence and source snapshots.

Measurement identity covers the checkpoint, both dataset identities,
resolved scientific settings, code provenance and every frozen numerical
record. Rendering artifacts/style are excluded. A render-only publisher
must retain the original measurement identity, values and code identity;
it never calls this builder to rederive an existing measurement.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.config import ModelAnalysisConfig, config_digest
from tools.ml_models.analysis.contracts import ArtifactRef, AvailabilityRecord, CodeIdentity
from tools.ml_models.analysis.model_measurement import ModelMeasurement
from tools.ml_models.analysis.summaries import ModelTrainingSummary


def model_summary(
    measured: ModelMeasurement,
    cfg: ModelAnalysisConfig,
    code: CodeIdentity,
    artifacts: tuple[ArtifactRef, ...],
    rendering_outputs: tuple[AvailabilityRecord, ...],
) -> Result[ModelTrainingSummary, str]:
    """Assemble one canonical analysis summary without inference, fitting or resampling."""
    try:
        if measured.config_digest != config_digest(cfg):
            return Err("model summary settings differ from the frozen measurement configuration")
        payload = {
            "schema_version": 1,
            "summary_kind": "MODEL_TRAINING_ANALYSIS",
            "training_dataset": asdict(measured.training_dataset),
            "evaluation_dataset": asdict(measured.evaluation_dataset),
            "checkpoint": asdict(measured.checkpoint),
            "config_digest": config_digest(cfg),
            "code": asdict(code),
            "splits": [asdict(capture.evidence) for capture in measured.captures],
            "baseline": asdict(measured.baseline),
            "generalization": [asdict(record) for record in measured.generalization],
            "development": asdict(measured.development) if measured.development else None,
            "training_figures": [asdict(record) for record in measured.training_figures],
            "task_figures": [asdict(record) for record in measured.task_figures],
            "generalization_figures": [
                asdict(record) for record in measured.generalization_figures
            ],
            "resources": [asdict(record) for record in measured.resources],
            "metric_reference": [asdict(record) for record in measured.reference],
            "source_snapshots": [
                {
                    "path": file.path,
                    "sha256": hashlib.sha256(file.data).hexdigest(),
                    "size_bytes": len(file.data),
                }
                for file in measured.snapshots
            ],
        }
        raw = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        measurement_id = hashlib.sha256(raw).hexdigest()
        outputs = {output.name: output for output in measured.outputs}
        if len(outputs) != len(measured.outputs):
            return Err("model measurement contains duplicate declared output names")
        history = next(
            (reference for reference in artifacts if reference.path == "source/history.jsonl"), None
        )
        outputs["training_history"] = AvailabilityRecord(
            name="training_history",
            status="AVAILABLE" if history else "UNAVAILABLE",
            reason=None
            if history
            else "No durable history file was captured; incomplete/missing history stays explicit",
        )
        outputs["training_baseline"] = AvailabilityRecord(
            name="training_baseline", status="AVAILABLE", required=True
        )
        for capture, generalized in zip(measured.captures, measured.generalization, strict=True):
            if (
                capture.evidence.dataset_hash,
                capture.evidence.dataset_manifest_hash,
                capture.evidence.task,
                capture.evidence.split,
            ) != (
                generalized.dataset_hash,
                generalized.dataset_manifest_hash,
                generalized.task,
                generalized.split,
            ):
                return Err("generalization snapshot is not aligned with its model split")
            for output in generalized.outputs:
                name = "generalization:" + generalized.split + ":" + output.name
                outputs[name] = AvailabilityRecord(
                    name=name, status=output.status, reason=output.reason, required=output.required
                )
        for resource in measured.resources:
            name = "resource:" + "x".join(str(value) for value in resource.image_shape)
            if name in outputs:
                return Err("model resource evidence repeats an input shape")
            outputs[name] = AvailabilityRecord(
                name=name,
                status="AVAILABLE" if resource.evidence else "UNAVAILABLE",
                reason=None
                if resource.evidence
                else resource.reason or "Resource measurement unavailable",
            )
        outputs["flight_qualification"] = AvailabilityRecord(
            name="flight_qualification",
            status="SKIPPED",
            reason="Individual-model evidence is not target-hardware or "
            "complete flight qualification",
        )
        seen: set[str] = set()
        for output in rendering_outputs:
            if output.name in seen or output.name in outputs:
                return Err("rendering output cannot replace a scientific availability record")
            if output.name not in ("figures", "visuals") and not output.name.startswith(
                ("model_figure:", "training_figure:", "prediction_visual:")
            ):
                return Err("unknown rendering availability namespace")
            seen.add(output.name)
            outputs[output.name] = output
        figures = outputs.get("figures")
        if figures is None or figures.status != "AVAILABLE" or not figures.required:
            return Err("model summary requires completed indexed figure rendering")
        artifact_kinds = {reference.path: reference.kind for reference in artifacts}
        required_charts = [
            (
                "figures/training/" + record.identifier,
                "training_figure:" + record.identifier,
                "FIGURE",
            )
            for record in measured.training_figures
        ]
        for family, records in (
            (measured.checkpoint.kind, measured.task_figures),
            ("generalization", measured.generalization_figures),
        ):
            required_charts.extend(
                (
                    "figures/" + family + "/" + record.identity.split + "/" + record.identifier,
                    "model_figure:"
                    + family
                    + ":"
                    + record.identity.split
                    + ":"
                    + record.identifier,
                    "FIGURE",
                )
                for record in records
            )
        for capture in measured.captures:
            for gallery in capture.previews.galleries:
                per_page = 4 if measured.checkpoint.kind == "classifier" else 1
                pages = max(1, (len(gallery.rows) + per_page - 1) // per_page)
                required_charts.extend(
                    (
                        "visuals/predictions/" + gallery.identifier + "_" + str(page + 1),
                        gallery.availability.name,
                        "VISUAL",
                    )
                    for page in range(pages)
                )
        if any(
            name not in outputs
            or any(artifact_kinds.get(path + "." + fmt) != kind for fmt in cfg.plot.formats)
            for path, name, kind in required_charts
        ):
            return Err(
                "model summary is missing required frozen chart artifacts or availability indexes"
            )
        unavailable_recipes = [
            ("training_figure:" + record.identifier, record.reason)
            for record in measured.training_figures
            if record.reason is not None
        ]
        for family, records in (
            (measured.checkpoint.kind, measured.task_figures),
            ("generalization", measured.generalization_figures),
        ):
            unavailable_recipes.extend(
                (
                    "model_figure:"
                    + family
                    + ":"
                    + record.identity.split
                    + ":"
                    + record.identifier,
                    record.reason,
                )
                for record in records
                if record.reason is not None
            )
        if any(
            outputs[name].status != "UNAVAILABLE" or outputs[name].reason != reason
            for name, reason in unavailable_recipes
        ):
            return Err("rendering cannot relabel an unavailable frozen chart recipe")
        warnings = tuple(
            dict.fromkeys(
                measured.warnings
                + tuple(
                    warning for record in measured.generalization for warning in record.warnings
                )
                + (
                    "Resource counts cover only supported dense Conv2d/Linear "
                    "operations, not total graph FLOPs or latency.",
                )
            )
        )
        return Ok(
            ModelTrainingSummary(
                measurement_id=measurement_id,
                training_dataset=measured.training_dataset,
                evaluation_dataset=measured.evaluation_dataset,
                checkpoint=measured.checkpoint,
                code=code,
                config_digest=config_digest(cfg),
                splits=tuple(capture.evidence for capture in measured.captures),
                artifacts=artifacts,
                outputs=tuple(outputs.values()),
                history=history,
                warnings=warnings,
            )
        )
    except (ValueError, TypeError, OverflowError) as exc:
        return Err(f"cannot assemble model/training summary: {exc}")
