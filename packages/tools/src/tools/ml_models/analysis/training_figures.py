"""Frozen history-chart coordinates without re-averaging, inference or checkpoint ranking.

Attempted optimization batches and captured sample-weighted epochs are
separate from canonical train/validation evaluations. Actual checkpoint
records supply best/last markers; an optional test evaluation contributes
one point only at its verified selected checkpoint. Raw zeros stay in the
recipes, even when a logarithmic renderer must explicitly omit them.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.contracts import CheckpointIdentity, MetricValue, SplitEvidence
from tools.ml_models.analysis.training import (
    EpochRecord,
    EvaluationRecord,
    StepRecord,
    TrainingHistory,
)

type AxisScale = Literal["linear", "log"]


@dataclass(frozen=True, slots=True)
class TrainingSeries:
    """Raw event coordinates and exposure counts, not independent observation support."""

    name: str
    x: tuple[float, ...]
    y: tuple[float | None, ...]
    n_exposures: int
    population: str
    points_only: bool = False
    exposure_unit: str = "IMAGE"


@dataclass(frozen=True, slots=True)
class TrainingMarker:
    """Actual captured checkpoint/state position, never a re-selected model score."""

    name: str
    x: float
    checkpoint_hash: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingFigure:
    """One standalone chart recipe with explicit scales, run state and unavailable reason."""

    identifier: str
    title: str
    x_label: str
    y_label: str
    series: tuple[TrainingSeries, ...]
    markers: tuple[TrainingMarker, ...]
    run_status: str
    x_scale: AxisScale = "linear"
    y_scale: AxisScale = "linear"
    reason: str | None = None
    warnings: tuple[str, ...] = ()


def _loss_metric(evidence: SplitEvidence, name: str) -> MetricValue | None:
    """Return a named captured value without inferring missing objective components."""
    return next((metric for metric in evidence.metrics if metric.name == name), None)


def _comparable(metrics: tuple[MetricValue, ...]) -> bool:
    """Require stable physical units, aggregation and operating threshold across captured states."""
    return (
        len(
            {
                (metric.unit, metric.aggregation, metric.threshold, metric.support.unit)
                for metric in metrics
            }
        )
        <= 1
    )


def training_figure_data(
    history: TrainingHistory,
    *,
    final_test: SplitEvidence | None = None,
    checkpoint: CheckpointIdentity | None = None,
) -> Result[tuple[TrainingFigure, ...], str]:
    """Freeze optimization/evaluation/telemetry figures from validated history evidence.

    Final-test evidence must name one actually selected checkpoint; only its
    single captured scalar point can be shown. No test tensor loading, loss
    re-reduction, interpolation, smoothing, resampling or model inference
    occurs. Incomplete histories retain their captured prefix and run status.
    """
    execution = history.execution
    selected = (
        checkpoint
        if checkpoint is not None
        else execution.selected.identity
        if execution.selected is not None
        else None
    )
    if final_test is not None and (
        selected is None
        or final_test.split != "test"
        or final_test.checkpoint_hash != selected.sha256
        or selected.step is None
        or selected.training_dataset_hash != execution.dataset_hash
        or selected.kind != execution.kind
        or selected.arch != execution.arch
        or final_test.task != execution.kind
    ):
        return Err(
            "final-test history point requires the actual selected checkpoint identity and step"
        )
    steps = tuple(record for record in history.records if isinstance(record, StepRecord))
    epochs = tuple(record for record in history.records if isinstance(record, EpochRecord))
    evaluations = tuple(
        record for record in history.records if isinstance(record, EvaluationRecord)
    )
    recorded_checkpoints = tuple(record.checkpoint.identity for record in evaluations) + tuple(
        record.identity
        for record in (execution.best, execution.last, execution.selected)
        if record is not None
    )
    if final_test is not None and selected not in recorded_checkpoints:
        return Err("final-test checkpoint was not recorded in this training history")
    if any(
        record.train.dataset_hash != execution.dataset_hash
        or record.validation.dataset_hash != execution.dataset_hash
        for record in evaluations
    ):
        return Err("training figures require captured evaluation dataset identity to match the run")
    markers: list[TrainingMarker] = []
    for name, recorded in (
        ("best checkpoint", execution.best),
        ("last evaluated checkpoint", execution.last),
    ):
        if recorded is not None and recorded.identity.step is not None:
            markers.append(
                TrainingMarker(name, float(recorded.identity.step), recorded.identity.sha256)
            )
    if execution.status != "RUNNING":
        markers.append(TrainingMarker("stopping state", float(execution.optimizer_step)))
    elif steps:
        markers.append(
            TrainingMarker("last captured running state", float(steps[-1].optimizer_step))
        )
    point_markers = tuple(markers)
    figures: list[TrainingFigure] = []
    warnings = history.warnings + (
        ("Incomplete training history; no completed execution is inferred.",)
        if execution.status != "COMPLETED"
        else ()
    )
    step_x = tuple(float(record.optimizer_step) for record in steps)
    epoch_x = tuple(float(record.optimizer_step) for record in epochs)
    evaluation_x = tuple(float(record.optimizer_step) for record in evaluations)
    exposure = sum(record.n_samples for record in steps)

    def figure(
        identifier: str,
        title: str,
        y_label: str,
        series: tuple[TrainingSeries, ...],
        *,
        x_scale: AxisScale = "linear",
        y_scale: AxisScale = "linear",
        reason: str = "Requested values were not captured or are inactive",
    ) -> TrainingFigure:
        """Build an exact recipe; missingness and logarithmic-domain handling stay explicit."""
        available = any(value is not None for row in series for value in row.y)
        return TrainingFigure(
            identifier,
            title,
            "Successful optimizer updates",
            y_label,
            series,
            point_markers,
            execution.status,
            x_scale,
            y_scale,
            None if available else reason,
            warnings,
        )

    batch_population = "attempted optimization batches; resampled/augmented image exposures"
    attempted = TrainingSeries(
        "optimization batch objective",
        step_x,
        tuple(record.loss.total for record in steps),
        exposure,
        batch_population,
    )
    optimization_scales: tuple[tuple[str, AxisScale, AxisScale], ...] = (
        ("optimization_loss_linear", "linear", "linear"),
        ("optimization_loss_semilog", "linear", "log"),
        ("optimization_loss_loglog", "log", "log"),
    )
    for identifier, x_scale, y_scale in optimization_scales:
        figures.append(
            figure(
                identifier,
                "Optimization objective per attempted batch",
                "Configured weighted objective",
                (attempted,),
                x_scale=x_scale,
                y_scale=y_scale,
            )
        )
    for name, step_values, epoch_values in (
        (
            "total",
            tuple(record.loss.total for record in steps),
            tuple(record.loss.total for record in epochs),
        ),
        (
            "bce",
            tuple(record.loss.bce for record in steps),
            tuple(record.loss.bce for record in epochs),
        ),
        (
            "focal",
            tuple(record.loss.focal for record in steps),
            tuple(record.loss.focal for record in epochs),
        ),
        (
            "dice",
            tuple(record.loss.dice for record in steps),
            tuple(record.loss.dice for record in epochs),
        ),
    ):
        if name != "total":
            figures.append(
                figure(
                    "optimization_component_" + name,
                    "Weighted optimization " + name + " component",
                    "Weighted " + name + " contribution",
                    (TrainingSeries(name, step_x, step_values, exposure, batch_population),),
                )
            )
        figures.append(
            figure(
                "optimization_epoch_" + name,
                "Captured sample-weighted epoch " + name,
                "Configured weighted objective"
                if name == "total"
                else "Weighted " + name + " contribution",
                (
                    TrainingSeries(
                        "captured weighted epoch",
                        epoch_x,
                        epoch_values,
                        sum(record.n_samples for record in epochs),
                        "sample-weighted optimization epoch records; no batch-mean re-averaging",
                    ),
                ),
            )
        )
    evaluation_series: list[TrainingSeries] = []
    objective_definitions: list[MetricValue] = []
    for name, population, captured in (
        (
            "canonical train",
            "canonical train images, repeated at captured checkpoint states",
            tuple(record.train for record in evaluations),
        ),
        (
            "validation",
            "validation images, repeated at captured checkpoint states",
            tuple(record.validation for record in evaluations),
        ),
    ):
        metrics = tuple(_loss_metric(evidence, "objective_loss") for evidence in captured)
        known = tuple(metric for metric in metrics if metric is not None)
        objective_definitions.extend(known)
        if not _comparable(known):
            return Err("captured evaluation objective definition changed across training history")
        evaluation_series.append(
            TrainingSeries(
                name,
                evaluation_x,
                tuple(metric.value if metric is not None else None for metric in metrics),
                sum(metric.support.n for metric in known),
                population,
            )
        )
    if final_test is not None and selected is not None and selected.step is not None:
        metric = _loss_metric(final_test, "objective_loss")
        if metric is not None:
            objective_definitions.append(metric)
        test_population = "one selected-checkpoint test evaluation" + (
            " on an external evaluation dataset"
            if final_test.dataset_hash != execution.dataset_hash
            else ""
        )
        evaluation_series.append(
            TrainingSeries(
                "selected-checkpoint test",
                (float(selected.step),),
                (metric.value if metric is not None else None,),
                metric.support.n if metric is not None else 0,
                test_population,
                points_only=True,
            )
        )
    if not _comparable(tuple(objective_definitions)):
        return Err(
            "canonical train/validation/test objectives have incompatible captured definitions"
        )
    evaluation_scales: tuple[tuple[str, AxisScale, AxisScale], ...] = (
        ("evaluation_loss_linear", "linear", "linear"),
        ("evaluation_loss_semilog", "linear", "log"),
        ("evaluation_loss_loglog", "log", "log"),
    )
    for identifier, x_scale, y_scale in evaluation_scales:
        figures.append(
            figure(
                identifier,
                "Comparable canonical evaluation objective",
                "Configured weighted objective",
                tuple(evaluation_series),
                x_scale=x_scale,
                y_scale=y_scale,
                reason="No canonical evaluation objective was captured",
            )
        )
    for component in ("bce", "focal", "dice"):
        component_series: list[TrainingSeries] = []
        definitions: list[MetricValue] = []
        for label, captured in (
            ("canonical train", tuple(record.train for record in evaluations)),
            ("validation", tuple(record.validation for record in evaluations)),
        ):
            values = tuple(
                _loss_metric(evidence, "objective_" + component) for evidence in captured
            )
            known = tuple(metric for metric in values if metric is not None)
            definitions.extend(known)
            component_series.append(
                TrainingSeries(
                    label,
                    evaluation_x,
                    tuple(metric.value if metric is not None else None for metric in values),
                    sum(metric.support.n for metric in known),
                    "captured canonical weighted objective components",
                )
            )
        if final_test is not None and selected is not None and selected.step is not None:
            metric = _loss_metric(final_test, "objective_" + component)
            if metric is not None:
                definitions.append(metric)
            component_series.append(
                TrainingSeries(
                    "selected-checkpoint test",
                    (float(selected.step),),
                    (metric.value if metric is not None else None,),
                    metric.support.n if metric is not None else 0,
                    "one selected-checkpoint weighted objective component",
                    points_only=True,
                )
            )
        if not _comparable(tuple(definitions)):
            return Err("captured canonical objective component definition changed")
        figures.append(
            figure(
                "evaluation_component_" + component,
                "Comparable canonical weighted " + component,
                "Weighted " + component + " contribution",
                tuple(component_series),
            )
        )
    names = sorted(
        {
            metric.name
            for record in evaluations
            for evidence in (record.train, record.validation)
            for metric in evidence.metrics
            if not metric.name.startswith("objective_")
        }
    )
    if not names:
        names = [execution.metric]
    for name in names:
        traces: list[TrainingSeries] = []
        all_known: list[MetricValue] = []
        for label, captured in (
            ("canonical train", tuple(record.train for record in evaluations)),
            ("validation", tuple(record.validation for record in evaluations)),
        ):
            values = tuple(_loss_metric(evidence, name) for evidence in captured)
            known = tuple(metric for metric in values if metric is not None)
            all_known.extend(known)
            traces.append(
                TrainingSeries(
                    label,
                    evaluation_x,
                    tuple(metric.value if metric is not None else None for metric in values),
                    sum(metric.support.n for metric in known),
                    "captured " + label + " metric populations",
                    exposure_unit=known[0].support.unit if known else "unavailable",
                )
            )
        if not _comparable(tuple(all_known)):
            return Err(f"captured metric {name} definition/threshold changed across history")
        unit = all_known[0].unit if all_known else "unavailable"
        figures.append(
            figure(
                "training_metric_" + name,
                "Captured " + name + " trajectory",
                name + " (" + unit + ")",
                tuple(traces),
            )
        )
        gaps: list[float | None] = []
        for record in evaluations:
            train_value, validation_value = (
                _loss_metric(record.train, name),
                _loss_metric(record.validation, name),
            )
            gaps.append(
                train_value.value - validation_value.value
                if train_value is not None
                and validation_value is not None
                and train_value.value is not None
                and validation_value.value is not None
                else None
            )
        if any(value is not None and not math.isfinite(value) for value in gaps):
            return Err("training metric gap exceeds finite numerical range")
        figures.append(
            figure(
                "development_gap_" + name,
                "Captured train-minus-validation " + name,
                "Train minus validation (" + unit + ")",
                (
                    TrainingSeries(
                        "train minus validation",
                        evaluation_x,
                        tuple(gaps),
                        len(evaluations),
                        "same captured checkpoint/task/dataset metric definitions",
                        exposure_unit="PAIRED_EVALUATION",
                    ),
                ),
            )
        )
    groups = len(steps[0].learning_rates) if steps else 0
    if any(len(record.learning_rates) != groups for record in steps):
        return Err("optimizer parameter-group layout changed within captured history")
    figures.append(
        figure(
            "learning_rates",
            "Recorded optimizer learning rates",
            "Learning rate",
            tuple(
                TrainingSeries(
                    "parameter group " + str(index),
                    step_x,
                    tuple(record.learning_rates[index] for record in steps),
                    len(steps),
                    "recorded attempted batch events",
                    exposure_unit="BATCH",
                )
                for index in range(groups)
            ),
        )
    )
    for identifier, title, unit, telemetry in (
        (
            "batch_duration",
            "Recorded attempted-batch duration",
            "s",
            tuple(record.duration_s for record in steps),
        ),
        (
            "batch_throughput",
            "Recorded attempted-batch throughput",
            "image exposures/s",
            tuple(record.throughput_images_s for record in steps),
        ),
        (
            "gradient_norm",
            "Recorded pre-update gradient norm",
            "gradient norm",
            tuple(record.gradient_norm for record in steps),
        ),
        (
            "optimizer_update_applied",
            "Recorded optimizer updates and AMP skips",
            "Update applied (0 or 1)",
            tuple(float(record.update_applied) for record in steps),
        ),
    ):
        figures.append(
            figure(
                identifier,
                title,
                unit,
                (
                    TrainingSeries(
                        title,
                        step_x,
                        telemetry,
                        len(steps),
                        "recorded attempted optimization batches",
                        exposure_unit="BATCH",
                    ),
                ),
            )
        )
    figures.append(
        figure(
            "amp_scale",
            "Recorded AMP scaler telemetry",
            "AMP scale",
            (
                TrainingSeries(
                    "before update",
                    step_x,
                    tuple(record.amp_scale_before for record in steps),
                    len(steps),
                    "recorded AMP events",
                    exposure_unit="BATCH",
                ),
                TrainingSeries(
                    "after update",
                    step_x,
                    tuple(record.amp_scale_after for record in steps),
                    len(steps),
                    "recorded AMP events",
                    exposure_unit="BATCH",
                ),
            ),
        )
    )
    for identifier, title, unit, durations in (
        (
            "epoch_duration",
            "Captured optimization epoch duration",
            "s",
            tuple(record.duration_s for record in epochs),
        ),
        (
            "epoch_throughput",
            "Captured optimization epoch throughput",
            "image exposures/s",
            tuple(record.throughput_images_s for record in epochs),
        ),
        (
            "evaluation_duration",
            "Captured train/validation evaluation duration",
            "s",
            tuple(record.duration_s for record in evaluations),
        ),
    ):
        positions = evaluation_x if identifier == "evaluation_duration" else epoch_x
        figures.append(
            figure(
                identifier,
                title,
                unit,
                (
                    TrainingSeries(
                        title,
                        positions,
                        durations,
                        len(durations),
                        "captured epoch/evaluation records",
                        exposure_unit="EVALUATION"
                        if identifier == "evaluation_duration"
                        else "EPOCH",
                    ),
                ),
            )
        )
    return Ok(tuple(figures))
