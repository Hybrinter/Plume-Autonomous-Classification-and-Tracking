"""Stratum, interval, baseline and heatmap recipes from frozen generalization evidence.

Intervals and stratum values are copied verbatim. No resampling, source
loading, model evaluation, baseline fitting or metric reduction occurs.
Missing metadata cohorts retain explicit labels and support; grouped
random holdouts are not presented as temporal or unseen-sensor proof.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.contracts import MetricSupport, SplitEvidence, StratumEvidence
from tools.ml_models.analysis.metrics.generalization import GeneralizationEvidence
from tools.ml_models.analysis.model_figures import (
    ModelFigure,
    ModelSeries,
    curve_figure,
    figure_identity,
    metric_figure,
    safe_token,
)


def generalization_figure_data(
    evidence: SplitEvidence,
    frozen: GeneralizationEvidence,
) -> Result[tuple[ModelFigure, ...], str]:
    """Copy a source-bound frozen generalization snapshot to standalone chart recipes."""
    if (evidence.dataset_hash, evidence.dataset_manifest_hash, evidence.task, evidence.split) != (
        frozen.dataset_hash,
        frozen.dataset_manifest_hash,
        frozen.task,
        frozen.split,
    ):
        return Err("generalization figure snapshot differs from the recorded split identity")
    identity = figure_identity(evidence)
    figures = [metric_figure(evidence, metric) for metric in frozen.metrics]
    figures = [
        ModelFigure(
            "grouped_" + figure.identifier,
            figure.identity,
            figure.title,
            figure.x_label,
            figure.y_label,
            figure.population,
            figure.series,
            figure.points,
            figure.kind,
            figure.x_categories,
            figure.y_categories,
            figure.matrix,
            figure.matrix_support,
            figure.reason,
            figure.notes + frozen.warnings,
            figure.matrix_range,
        )
        for figure in figures
    ]
    strata_by_name: dict[str, list[StratumEvidence]] = defaultdict(list)
    for stratum in frozen.strata:
        strata_by_name[stratum.name].append(stratum)
    for name, strata in sorted(strata_by_name.items()):
        ordered = sorted(
            strata, key=lambda stratum: (stratum.value is not None, stratum.value or "")
        )
        labels = tuple(
            repr(stratum.value) if stratum.value is not None else "(recorded metadata missing)"
            for stratum in ordered
        )
        names = sorted({metric.name for stratum in ordered for metric in stratum.metrics})
        for metric_name in names:
            metrics = tuple(
                next((metric for metric in stratum.metrics if metric.name == metric_name), None)
                for stratum in ordered
            )
            definitions = {
                (metric.unit, metric.aggregation, metric.support.unit, metric.threshold)
                for metric in metrics
                if metric is not None
            }
            if len(definitions) > 1:
                return Err("frozen stratum metrics mix incompatible definitions")
            first = next((metric for metric in metrics if metric is not None), None)
            if first is None:
                continue
            values = tuple(metric.value if metric is not None else None for metric in metrics)
            counts = tuple(metric.support.n if metric is not None else 0 for metric in metrics)
            lower = tuple(
                metric.interval.lower if metric is not None and metric.interval else None
                for metric in metrics
            )
            upper = tuple(
                metric.interval.upper if metric is not None and metric.interval else None
                for metric in metrics
            )
            identifier = "stratum_" + safe_token(name) + "_" + metric_name
            base = ModelFigure(
                identifier,
                identity,
                metric_name.replace("_", " ") + " by " + name,
                name,
                metric_name + " (" + first.unit + ")",
                evidence.split + "; frozen strata; related variants remain correlated",
            )
            pages = (len(ordered) + 5) // 6
            for page in range(pages):
                start, stop = page * 6, min(len(ordered), (page + 1) * 6)
                selected_metrics = metrics[start:stop]
                intervals = sorted(
                    {
                        (metric.interval.confidence, metric.interval.method)
                        for metric in selected_metrics
                        if metric is not None and metric.interval is not None
                    }
                )
                page_notes = (
                    first.aggregation,
                    "Frozen intervals: "
                    + "; ".join(f"{confidence:g} {method}" for confidence, method in intervals)
                    if intervals
                    else "Intervals unavailable for these recorded strata",
                    "Null points/absent interval bars are unavailable, not zero; "
                    "see frozen metric and valid-replicate audit.",
                )
                series = ModelSeries(
                    metric_name,
                    tuple(float(index) for index in range(stop - start)),
                    values[start:stop],
                    MetricSupport(unit=first.support.unit, n=sum(counts[start:stop])),
                    "POINT",
                    lower[start:stop],
                    upper[start:stop],
                    counts[start:stop],
                )
                figures.append(
                    replace(
                        base,
                        identifier=identifier
                        if pages == 1
                        else identifier + "_page_" + str(page + 1),
                        title=base.title
                        if pages == 1
                        else base.title + f" (page {page + 1}/{pages})",
                        series=(series,),
                        x_categories=labels[start:stop],
                        notes=page_notes,
                        reason=None
                        if any(value is not None for value in values[start:stop])
                        else "Metric unavailable in every recorded stratum on this page",
                    )
                )
            if name in (
                "gsd_lateral_by_truth_image_area_px",
                "gsd_lateral_by_truth_component_area_px",
            ):
                cells: dict[tuple[str, str], tuple[float | None, int]] = {}
                for stratum, metric in zip(ordered, metrics, strict=True):
                    parts = (
                        stratum.value.split("|")
                        if stratum.value is not None
                        else ["(missing GSD/size)", "(missing GSD/size)"]
                    )
                    if len(parts) != 2:
                        return Err(
                            "frozen GSD-size stratum does not record exactly two category axes"
                        )
                    key = (parts[0], parts[1])
                    if key in cells:
                        return Err("frozen GSD-size strata repeat a heatmap cell")
                    cells[key] = (
                        metric.value if metric else None,
                        metric.support.n if metric else 0,
                    )
                x_labels = tuple(sorted({key[0] for key in cells}))
                y_labels = tuple(sorted({key[1] for key in cells}))
                matrix = tuple(
                    tuple(cells.get((x, y), (None, 0))[0] for x in x_labels) for y in y_labels
                )
                support = tuple(
                    tuple(cells.get((x, y), (None, 0))[1] for x in x_labels) for y in y_labels
                )
                matrix_values = tuple(value for row in matrix for value in row if value is not None)
                bounds = (min(matrix_values), max(matrix_values)) if matrix_values else None
                base_heatmap = ModelFigure(
                    "heatmap_" + safe_token(name) + "_" + metric_name,
                    identity,
                    metric_name.replace("_", " ") + " by GSD and truth size",
                    "Recorded GSD category (m)",
                    "Recorded truth area category (pixel)",
                    evidence.split + "; frozen cell estimates; annotations show support",
                    kind="MATRIX",
                    x_categories=x_labels,
                    y_categories=y_labels,
                    matrix=matrix,
                    matrix_support=support,
                    reason=None
                    if any(value is not None for row in matrix for value in row)
                    else "No available GSD-size cell estimates",
                    notes=(
                        "Intervals are retained in the companion stratum figure/table; "
                        "absent cells are not zero.",
                    ),
                    matrix_range=bounds,
                )
                row_pages, column_pages = (len(y_labels) + 5) // 6, (len(x_labels) + 5) // 6
                for row_page in range(row_pages):
                    for column_page in range(column_pages):
                        first_row, first_column = row_page * 6, column_page * 6
                        page_matrix = tuple(
                            row[first_column : first_column + 6]
                            for row in matrix[first_row : first_row + 6]
                        )
                        page_support = tuple(
                            row[first_column : first_column + 6]
                            for row in support[first_row : first_row + 6]
                        )
                        suffix = f"_page_{row_page + 1}_{column_page + 1}"
                        figures.append(
                            replace(
                                base_heatmap,
                                identifier=base_heatmap.identifier
                                if row_pages == column_pages == 1
                                else base_heatmap.identifier + suffix,
                                x_categories=x_labels[first_column : first_column + 6],
                                y_categories=y_labels[first_row : first_row + 6],
                                matrix=page_matrix,
                                matrix_support=page_support,
                                reason=None
                                if any(value is not None for row in page_matrix for value in row)
                                else "No available GSD-size cell estimates on this page",
                                title=base_heatmap.title
                                if row_pages == column_pages == 1
                                else base_heatmap.title
                                + f" (tile {row_page + 1}/{row_pages}, "
                                + f"{column_page + 1}/{column_pages})",
                            )
                        )
    baseline_metrics = {metric.name: metric for metric in frozen.baseline_metrics}
    for metric in frozen.metrics:
        baseline = baseline_metrics.get(metric.name)
        if baseline is None:
            continue
        figures.append(
            ModelFigure(
                "baseline_" + metric.name,
                identity,
                "Frozen training-derived baseline comparison",
                "Policy",
                metric.name + " (" + metric.unit + ")",
                evidence.split + "; baseline was fitted on canonical training data only",
                (
                    ModelSeries(
                        metric.name,
                        (0.0, 1.0),
                        (metric.value, baseline.value),
                        metric.support,
                        "POINT",
                        point_support=(metric.support.n, baseline.support.n),
                    ),
                ),
                x_categories=("selected checkpoint", "fixed train-derived baseline"),
                reason=None
                if metric.value is not None or baseline.value is not None
                else "Both model and baseline values unavailable",
                notes=(metric.aggregation, baseline.aggregation),
            )
        )
    baseline_evidence = SplitEvidence(
        task=evidence.task,
        split=evidence.split,
        dataset_hash=evidence.dataset_hash,
        dataset_manifest_hash=evidence.dataset_manifest_hash,
        checkpoint_hash=evidence.checkpoint_hash,
        curves=frozen.baseline_curves,
        support=evidence.support,
    )
    for curve in frozen.baseline_curves:
        figure = curve_figure(
            baseline_evidence,
            curve.name,
            style="PRE" if curve.name == "precision_recall" else "LINE",
        )
        figures.append(
            ModelFigure(
                "baseline_curve_" + figure.identifier,
                identity,
                "Fixed train-derived baseline " + figure.title,
                figure.x_label,
                figure.y_label,
                figure.population,
                figure.series,
                reason=figure.reason,
                notes=figure.notes,
            )
        )
    for output in frozen.outputs:
        if output.status != "AVAILABLE":
            figures.append(
                ModelFigure(
                    "coverage_" + safe_token(output.name),
                    identity,
                    "Generalization coverage: " + output.name,
                    "Recorded metadata",
                    "Applicability",
                    evidence.split,
                    reason=output.reason or "Requested generalization evidence unavailable",
                )
            )
    return Ok(tuple(figures))
