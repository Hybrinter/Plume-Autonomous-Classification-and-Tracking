"""Selected-checkpoint measurement policy for the evidence-first model workflow.

The caller loads and verifies immutable run/model/dataset inputs first.
Canonical train and validation are development populations; test is measured
once only under explicit final_test after checkpoint and scoring settings
are fixed. Original training rows alone fit the baseline. Shared bounded
capture budgets cover every split, including external-baseline scalars.
Every reduction/selection recipe is frozen here for later pure rendering.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from flight.libs.types import Err, Ok, Result
from pydantic import TypeAdapter

from tools.ml_models.analysis.artifacts import BundleFile
from tools.ml_models.analysis.capture import BoundedCaptureSink, CaptureRow
from tools.ml_models.analysis.classifier_figures import classifier_figure_data
from tools.ml_models.analysis.config import (
    CaptureConfig,
    EvaluationConfig,
    ModelAnalysisConfig,
    config_digest,
)
from tools.ml_models.analysis.contracts import (
    AvailabilityRecord,
    CheckpointIdentity,
    DatasetIdentity,
    Split,
    SplitEvidence,
)
from tools.ml_models.analysis.cost import ResourceEvidence, measure_resources
from tools.ml_models.analysis.dataset_previews import display_channels
from tools.ml_models.analysis.evaluate import evaluate_split
from tools.ml_models.analysis.generalization_figures import generalization_figure_data
from tools.ml_models.analysis.metrics.generalization import (
    BaselineSpec,
    DevelopmentEvidence,
    GeneralizationEvidence,
    MetricReference,
    development_gaps,
    fit_baseline,
    measure_generalization,
    metric_reference,
)
from tools.ml_models.analysis.model_figures import ModelFigure, validate_figure_rows
from tools.ml_models.analysis.prediction_selections import prediction_gallery_data, prediction_key
from tools.ml_models.analysis.segmentation_figures import segmentation_figure_data
from tools.ml_models.analysis.training import TrainingHistory
from tools.ml_models.analysis.training_figures import TrainingFigure, training_figure_data
from tools.ml_models.analysis.visuals.predictions import PredictionPreview, PredictionPreviewCapture
from tools.ml_models.dataset.manifest import DatasetManifest

if TYPE_CHECKING:
    from torch import nn

    from tools.ml_models.train.losses import PlumeLoss


@dataclass(frozen=True, slots=True)
class ModelInputs:
    """Verified selected model and source snapshots supplied by the mechanical input loader."""

    history: TrainingHistory
    checkpoint: CheckpointIdentity
    training_dataset: DatasetIdentity
    evaluation_dataset: DatasetIdentity
    training_root: Path
    evaluation_root: Path
    training_manifest: DatasetManifest
    evaluation_manifest: DatasetManifest
    model: nn.Module
    objective: PlumeLoss
    snapshots: tuple[BundleFile, ...]


@dataclass(frozen=True, slots=True)
class ModelCapture:
    """Complete scalar split evidence plus bounded immutable preview/prediction bytes."""

    evidence: SplitEvidence
    rows: tuple[CaptureRow, ...]
    files: tuple[BundleFile, ...]
    previews: PredictionPreviewCapture


@dataclass(frozen=True, slots=True)
class ModelResource:
    """One selected-model/input-shape resource result or explicit unavailability."""

    image_shape: tuple[int, ...]
    evidence: ResourceEvidence | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class ModelMeasurement:
    """Frozen scientific outputs; no model/tensor handle is handed to the publisher."""

    training_dataset: DatasetIdentity
    evaluation_dataset: DatasetIdentity
    checkpoint: CheckpointIdentity
    config_digest: str
    captures: tuple[ModelCapture, ...]
    baseline_capture: ModelCapture | None
    baseline: BaselineSpec
    generalization: tuple[GeneralizationEvidence, ...]
    development: DevelopmentEvidence | None
    training_figures: tuple[TrainingFigure, ...]
    task_figures: tuple[ModelFigure, ...]
    generalization_figures: tuple[ModelFigure, ...]
    resources: tuple[ModelResource, ...]
    reference: tuple[MetricReference, ...]
    snapshots: tuple[BundleFile, ...]
    outputs: tuple[AvailabilityRecord, ...]
    warnings: tuple[str, ...]


def _capture(
    inputs: ModelInputs,
    cfg: ModelAnalysisConfig,
    workspace: Path,
    split: Split,
    capture_cfg: CaptureConfig,
    *,
    baseline_only: bool = False,
) -> Result[ModelCapture, str]:
    """Evaluate one fixed split once, then freeze the exact successful bounded capture bytes."""
    root = inputs.training_root if baseline_only else inputs.evaluation_root
    manifest = inputs.training_manifest if baseline_only else inputs.evaluation_manifest
    name = "baseline_train" if baseline_only else split
    destination = workspace / name
    sink = BoundedCaptureSink.create(destination, capture_cfg, dataset=root)
    if isinstance(sink, Err):
        return sink
    measured = evaluate_split(
        inputs.model,
        root,
        manifest,
        EvaluationConfig(
            kind=inputs.checkpoint.kind,
            split=split,
            batch_size=cfg.batch_size,
            device=cfg.device,
            score=cfg.score,
            capture=capture_cfg,
        ),
        objective=inputs.objective,
        capture=sink.value,
    )
    if isinstance(measured, Err):
        sink.value.abort(measured.error)
        sink.value.close()
        return measured
    closed = sink.value.close()
    if isinstance(closed, Err):
        return closed
    try:
        rows = tuple(
            TypeAdapter(CaptureRow).validate_json(line)
            for line in (destination / "rows.jsonl").read_bytes().splitlines()
        )
        files: list[BundleFile] = []
        references = []
        for reference in sink.value.references():
            data = destination.joinpath(*reference.path.split("/")).read_bytes()
            if (
                len(data) != reference.size_bytes
                or hashlib.sha256(data).hexdigest() != reference.sha256
            ):
                return Err("closed prediction capture changed before freezing")
            path = "capture/" + name + "/" + reference.path
            files.append(BundleFile(path, data))
            references.append(
                replace(reference, path=path, kind="REFERENCE", rows=None, population=None)
                if reference.path.startswith("previews/")
                else replace(reference, path=path)
            )
        evidence = replace(
            measured.value, checkpoint_hash=inputs.checkpoint.sha256, artifacts=tuple(references)
        )
        checked = validate_figure_rows(evidence, rows)
        if isinstance(checked, Err):
            return checked
        selected = prediction_gallery_data(
            evidence, rows, capture_cfg if baseline_only else cfg.capture
        )
        if isinstance(selected, Err):
            return selected
        mapping, label = display_channels(tuple(manifest.band_names))
        files_by_path = {file.path: file for file in files}
        previews: list[PredictionPreview] = []
        preview_files: list[BundleFile] = []
        selected_rows = {
            prediction_key(row): row for gallery in selected.value for row in gallery.rows
        }
        for key, row in sorted(selected_rows.items()):
            path = "capture/" + name + "/previews/" + key + ".npz"
            file = files_by_path.get(path)
            if file is not None:
                previews.append(
                    PredictionPreview(
                        row,
                        path,
                        hashlib.sha256(file.data).hexdigest(),
                        len(file.data),
                        mapping,
                        label,
                    )
                )
                preview_files.append(file)
        frozen_previews = PredictionPreviewCapture(
            tuple(previews), tuple(preview_files), selected.value
        )
        return Ok(ModelCapture(evidence, rows, tuple(files), frozen_previews))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
        return Err(f"cannot freeze successful prediction capture: {exc}")


def measure_model(
    inputs: ModelInputs,
    cfg: ModelAnalysisConfig,
    workspace: Path,
) -> Result[ModelMeasurement, str]:
    """Measure the approved fixed cohorts, freeze reductions, and never fit on held-out data."""
    try:
        if (
            inputs.checkpoint.kind != inputs.history.execution.kind
            or inputs.checkpoint.arch != inputs.history.execution.arch
            or inputs.checkpoint.training_dataset_hash != inputs.training_dataset.content_hash
            or inputs.training_dataset.content_hash != inputs.training_manifest.dataset_hash
            or inputs.evaluation_dataset.content_hash != inputs.evaluation_manifest.dataset_hash
        ):
            return Err("model measurement input snapshot identities disagree")
        if cfg.checkpoint not in ("best", "last"):
            return Err("model measurement requires an explicit best or last checkpoint selector")
        recorded = (
            inputs.history.execution.best
            if cfg.checkpoint == "best"
            else inputs.history.execution.last
        )
        if recorded is None or inputs.checkpoint != recorded.identity:
            return Err(
                "model measurement requires the actual recorded checkpoint "
                "named by the frozen selector"
            )
        external = inputs.training_dataset != inputs.evaluation_dataset
        remaining_bytes = cfg.capture.max_capture_bytes
        remaining_images = cfg.capture.max_preview_images
        captures: list[ModelCapture] = []
        outputs: list[AvailabilityRecord] = []
        baseline_capture: ModelCapture | None = None
        if external:
            baseline_cfg = replace(
                cfg.capture, retention="COMPACT", max_preview_images=0, examples_per_family=0
            )
            original = _capture(inputs, cfg, workspace, "train", baseline_cfg, baseline_only=True)
            if isinstance(original, Err):
                return original
            baseline_capture = original.value
            remaining_bytes -= sum(len(file.data) for file in original.value.files)
            baseline_rows = original.value.rows
        else:
            baseline_rows = ()
        active: tuple[Split, ...] = ("train", "val", "test") if cfg.final_test else ("train", "val")
        if not any(
            shard.task == inputs.checkpoint.kind and shard.split in active and shard.n > 0
            for shard in inputs.evaluation_manifest.shards
        ):
            return Err(
                "no eligible evaluation cohorts; request --final-test explicitly "
                "for a test-only external dataset"
            )
        for split in active:
            present = any(
                shard.task == inputs.checkpoint.kind and shard.split == split and shard.n > 0
                for shard in inputs.evaluation_manifest.shards
            )
            if not present:
                if not external or split == "test":
                    return Err(f"requested {split} evaluation has no eligible recorded task rows")
                outputs.append(
                    AvailabilityRecord(
                        name="evaluation:" + split,
                        status="UNAVAILABLE",
                        reason="External evaluation dataset has no eligible rows "
                        "for this recorded split",
                    )
                )
                continue
            if remaining_bytes < 1:
                return Err(
                    "shared model capture byte budget exhausted before required scalar evaluation"
                )
            capture_cfg = replace(
                cfg.capture,
                max_capture_bytes=remaining_bytes,
                max_preview_images=remaining_images,
                examples_per_family=min(cfg.capture.examples_per_family, remaining_images),
            )
            captured = _capture(inputs, cfg, workspace, split, capture_cfg)
            if isinstance(captured, Err):
                return captured
            captures.append(captured.value)
            remaining_bytes -= sum(len(file.data) for file in captured.value.files)
            remaining_images -= sum(
                file.path.startswith("capture/" + split + "/previews/")
                and file.path.endswith(".npz")
                for file in captured.value.files
            )
            outputs.append(
                AvailabilityRecord(name="evaluation:" + split, status="AVAILABLE", required=True)
            )
            if not external and split == "train":
                baseline_rows = captured.value.rows
        if not captures:
            return Err(
                "no eligible evaluation cohorts; request --final-test explicitly "
                "for a test-only external dataset"
            )
        fitted = fit_baseline(baseline_rows)
        if isinstance(fitted, Err):
            return fitted
        baseline = fitted.value
        generalization: list[GeneralizationEvidence] = []
        figures: list[ModelFigure] = []
        generalized_figures: list[ModelFigure] = []
        for capture in captures:
            task_figures = (
                classifier_figure_data(capture.evidence, capture.rows)
                if inputs.checkpoint.kind == "classifier"
                else segmentation_figure_data(capture.evidence, capture.rows)
            )
            if isinstance(task_figures, Err):
                return task_figures
            figures.extend(task_figures.value)
            generalized = measure_generalization(
                capture.rows, cfg.score, cfg.generalization, baseline=baseline
            )
            if isinstance(generalized, Err):
                return generalized
            generalization.append(generalized.value)
            strata_figures = generalization_figure_data(capture.evidence, generalized.value)
            if isinstance(strata_figures, Err):
                return strata_figures
            generalized_figures.extend(strata_figures.value)
        by_split = {capture.evidence.split: capture for capture in captures}
        development: DevelopmentEvidence | None = None
        if not external and "train" in by_split and "val" in by_split:
            gaps = development_gaps(
                by_split["train"].rows,
                by_split["val"].rows,
                cfg.score,
                checkpoint_hash=inputs.checkpoint.sha256,
            )
            if isinstance(gaps, Err):
                return gaps
            development = gaps.value
        outputs.append(
            AvailabilityRecord(
                name="development_gaps",
                status="AVAILABLE" if development else "UNAVAILABLE",
                reason=None
                if development
                else "External evaluation splits are evaluation-only, "
                "not original training/development cohorts",
            )
        )
        test = by_split.get("test")
        history = training_figure_data(
            inputs.history,
            final_test=test.evidence if test else None,
            checkpoint=inputs.checkpoint,
        )
        if isinstance(history, Err):
            return history
        resources: list[ModelResource] = []
        shapes = sorted(
            {
                (shard.height, shard.width)
                for shard in inputs.evaluation_manifest.shards
                if shard.task == inputs.checkpoint.kind and shard.split in by_split
            }
        )
        for height, width in shapes:
            for batch in sorted({1, cfg.batch_size}):
                image_shape = (batch, len(inputs.evaluation_dataset.band_names), height, width)
                measured = measure_resources(inputs.model, image_shape)
                resources.append(
                    ModelResource(
                        image_shape,
                        measured.value if isinstance(measured, Ok) else None,
                        measured.error if isinstance(measured, Err) else None,
                    )
                )
        outputs.append(
            AvailabilityRecord(
                name="final_test",
                status="AVAILABLE" if cfg.final_test else "SKIPPED",
                reason=None
                if cfg.final_test
                else "Test inference was not requested; checkpoint/settings are "
                "frozen before explicit final-test analysis",
            )
        )
        warnings = (
            (
                (
                    "External dataset splits are evaluation-only; no training or "
                    "transfer guarantee is implied.",
                )
                if external
                else ()
            )
            + inputs.history.warnings
            + inputs.history.execution.warnings
        )
        names = {metric.name for capture in captures for metric in capture.evidence.metrics}
        names.update(
            metric.name
            for capture in captures
            for stratum in capture.evidence.strata
            for metric in stratum.metrics
        )
        names.update(
            metric.name for capture in captures for row in capture.rows for metric in row.metrics
        )
        for record in generalization:
            names.update(metric.name for metric in (*record.metrics, *record.baseline_metrics))
            names.update(metric.name for stratum in record.strata for metric in stratum.metrics)
        if development is not None:
            names.update(gap.name for gap in development.gaps)
        references = metric_reference(tuple(sorted(names)))
        return Ok(
            ModelMeasurement(
                inputs.training_dataset,
                inputs.evaluation_dataset,
                inputs.checkpoint,
                config_digest(cfg),
                tuple(captures),
                baseline_capture,
                baseline,
                tuple(generalization),
                development,
                history.value,
                tuple(figures),
                tuple(generalized_figures),
                tuple(resources),
                references,
                inputs.snapshots,
                tuple(outputs),
                warnings,
            )
        )
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
        return Err(f"model measurement failed before publication: {exc}")
