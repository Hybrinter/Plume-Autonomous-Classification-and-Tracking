"""Frozen model-chart recipes and exact captured-row display reductions.

Coordinates, support, identities and missingness are retained separately
from presentation. Curve recipes copy captured coordinates without scoring;
ECDFs reduce scalar evidence once with complete tie grouping. Rendering
must consume these recipes, never call this module to remeasure a bundle.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.contracts import (
    CurveEvidence,
    MetricSupport,
    MetricValue,
    Split,
    SplitEvidence,
    SupportUnit,
    Task,
)

type DrawStyle = Literal["LINE", "PRE", "POST", "BAR", "POINT"]
type ModelFigureKind = Literal["SERIES", "MATRIX"]


@dataclass(frozen=True, slots=True)
class FigureIdentity:
    """Scientific source identity; a renderer cannot replace any of these fields."""

    dataset_hash: str
    dataset_manifest_hash: str | None
    checkpoint_hash: str | None
    task: Task
    split: Split


@dataclass(frozen=True, slots=True)
class ModelSeries:
    """Exact points and support, optional frozen intervals, and explicit draw convention."""

    name: str
    x: tuple[float, ...]
    y: tuple[float | None, ...]
    support: MetricSupport
    style: DrawStyle = "LINE"
    lower: tuple[float | None, ...] = ()
    upper: tuple[float | None, ...] = ()
    point_support: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelPoint:
    """One captured operating point, never an interpolated or fitted threshold."""

    name: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class ModelFigure:
    """Standalone recipe with source identity, population, coordinates and availability.

    Matrix rows follow y_categories and columns follow x_categories. Null
    cells and points remain missing. Intervals need not contain the point
    estimate; their absolute endpoints must not be re-centered by rendering.
    ``x_range``/``y_range`` are display bounds only: each side is a finite
    bound or None for the data-driven side; frozen coordinates never change.
    """

    identifier: str
    identity: FigureIdentity
    title: str
    x_label: str
    y_label: str
    population: str
    series: tuple[ModelSeries, ...] = ()
    points: tuple[ModelPoint, ...] = ()
    kind: ModelFigureKind = "SERIES"
    x_categories: tuple[str, ...] = ()
    y_categories: tuple[str, ...] = ()
    matrix: tuple[tuple[float | None, ...], ...] = ()
    matrix_support: tuple[tuple[int, ...], ...] = ()
    reason: str | None = None
    notes: tuple[str, ...] = ()
    matrix_range: tuple[float, float] | None = None
    x_range: tuple[float | None, float | None] | None = None
    y_range: tuple[float | None, float | None] | None = None


def figure_identity(evidence: SplitEvidence) -> FigureIdentity:
    """Copy the caller-bound split identity without resolving a model or dataset."""
    return FigureIdentity(
        evidence.dataset_hash,
        evidence.dataset_manifest_hash,
        evidence.checkpoint_hash,
        evidence.task,
        evidence.split,
    )


def safe_token(value: str) -> str:
    """Produce a stable file token while keeping the exact display value elsewhere."""
    stem = "".join(
        character if character.isascii() and character.isalnum() else "_" for character in value
    )
    return stem.lower().strip("_")[:48] + "_" + hashlib.sha256(value.encode()).hexdigest()[:8]


def captured_values(row: CaptureRow) -> dict[str, float | None]:
    """Read scalar values without manufacturing a missing annotation or denominator."""
    return {metric.name: metric.value for metric in row.metrics}


def validate_figure_rows(
    evidence: SplitEvidence,
    rows: tuple[CaptureRow, ...],
) -> Result[None, str]:
    """Require complete unique canonical scalar capture aligned with the frozen split."""
    if evidence.support.unit != "IMAGE" or evidence.support.n != len(rows):
        return Err("figure scalar capture must cover the complete recorded image cohort")
    identities = {
        (row.key.dataset_hash, row.dataset_manifest_hash, row.key.task, row.key.split)
        for row in rows
    }
    expected = (
        evidence.dataset_hash,
        evidence.dataset_manifest_hash,
        evidence.task,
        evidence.split,
    )
    if identities and identities != {expected}:
        return Err(
            "figure capture task/split/dataset/manifest identity differs from split evidence"
        )
    canonical = {(row.key.tile_id, row.bin_id, row.gsd_m, row.key.spatial_shard) for row in rows}
    if len({row.key for row in rows}) != len(rows) or len(canonical) != len(rows):
        return Err("figure capture contains duplicate canonical variants")
    return Ok(None)


def ecdf_series(
    name: str,
    values: tuple[float, ...],
    *,
    unit: SupportUnit = "IMAGE",
) -> ModelSeries:
    """Freeze a right-continuous ECDF with complete ties, including zero-support series."""
    counts = Counter(values)
    ordered = sorted(counts)
    total = 0
    fractions: list[float] = []
    for value in ordered:
        total += counts[value]
        fractions.append(total / len(values))
    return ModelSeries(
        name,
        tuple(ordered),
        tuple(fractions),
        MetricSupport(unit=unit, n=len(values)),
        "POST",
    )


def distribution_figure(
    evidence: SplitEvidence,
    identifier: str,
    title: str,
    x_label: str,
    populations: tuple[tuple[str, tuple[float, ...]], ...],
    *,
    population: str,
    unit: SupportUnit = "IMAGE",
    notes: tuple[str, ...] = (),
) -> ModelFigure:
    """Reduce only caller-selected frozen scalar populations, retaining empty cohorts."""
    series = tuple(ecdf_series(name, values, unit=unit) for name, values in populations)
    return ModelFigure(
        identifier,
        figure_identity(evidence),
        title,
        x_label,
        "Empirical cumulative fraction",
        population,
        series,
        reason=None if any(values for _, values in populations) else "No eligible captured values",
        notes=notes,
    )


def curve_figure(
    evidence: SplitEvidence,
    name: str,
    *,
    points: tuple[ModelPoint, ...] = (),
    baseline: ModelSeries | None = None,
    style: DrawStyle = "LINE",
    reason: str | None = None,
    x_range: tuple[float | None, float | None] | None = None,
    y_range: tuple[float | None, float | None] | None = None,
) -> ModelFigure:
    """Copy an existing curve verbatim, with captured method and approximation notes."""
    curve = next((item for item in evidence.curves if item.name == name), None)
    if curve is None:
        return ModelFigure(
            name,
            figure_identity(evidence),
            "Captured " + name.replace("_", " "),
            name,
            "Captured value",
            evidence.split + " eligible recorded cohort",
            reason=reason or "Requested curve was not captured",
        )
    series: tuple[ModelSeries, ...] = (ModelSeries(name, curve.x, curve.y, curve.support, style),)
    if baseline is not None:
        series += (baseline,)
    return ModelFigure(
        name,
        figure_identity(evidence),
        ("Approximate histogram " if curve.method == "HISTOGRAM" else "Captured ")
        + name.replace("_", " "),
        curve.x_name + " (" + curve.x_unit + ")",
        curve.y_name + " (" + curve.y_unit + ")",
        evidence.split + " " + curve.support.unit.lower() + " cohort",
        series,
        points,
        reason=None
        if any(value is not None for value in curve.y)
        else reason or "No eligible captured values for this curve",
        notes=curve.notes,
        x_range=x_range,
        y_range=y_range,
    )


def metric_figure(evidence: SplitEvidence, metric: MetricValue) -> ModelFigure:
    """Copy one scalar and absolute interval endpoints; never invent unsupported uncertainty."""
    interval = metric.interval
    return ModelFigure(
        "metric_" + metric.name,
        figure_identity(evidence),
        "Captured " + metric.name.replace("_", " "),
        "Recorded split",
        metric.name + " (" + metric.unit + ")",
        evidence.split + "; " + metric.aggregation,
        (
            ModelSeries(
                metric.name,
                (0.0,),
                (metric.value,),
                metric.support,
                "POINT",
                (interval.lower,) if interval else (),
                (interval.upper,) if interval else (),
            ),
        ),
        x_categories=(evidence.split,),
        reason=metric.reason,
        notes=(f"Operating threshold: {metric.threshold!r}",)
        if metric.threshold is not None
        else (),
    )


def curve_by_name(curves: tuple[CurveEvidence, ...], name: str) -> CurveEvidence | None:
    """Return a captured coordinate record without recomputing missing curves."""
    return next((curve for curve in curves if curve.name == name), None)
