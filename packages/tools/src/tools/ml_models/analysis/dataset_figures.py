"""Exact plot coordinates derived once from frozen dataset measurements.

These recipes never read source tensors or re-fit baselines. ECDF coordinates
group complete ties with equal canonical-image or component support; pixel
histograms and correlations pass through captured values unchanged.
Unavailable populations carry reasons instead of artificial zeros.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from tools.ml_models.analysis.contracts import MetricSupport, Split, SupportUnit
from tools.ml_models.analysis.dataset import DatasetMeasurement, DatasetSample

type FigureKind = Literal["BAR", "ECDF", "HISTOGRAM", "MATRIX"]
type Coordinate = str | float | int


@dataclass(frozen=True, slots=True)
class FigureSeries:
    """Frozen coordinates with support; null y values remain unavailable."""

    name: str
    x: tuple[Coordinate, ...]
    y: tuple[float | int | None, ...]
    support: MetricSupport


@dataclass(frozen=True, slots=True)
class DatasetFigure:
    """One standalone chart recipe with explicit population and units.

    Histogram x values are bin edges, one longer than the count vector.
    Matrix rows/columns follow ``matrix_labels`` exactly. ``reason`` declares
    an unavailable requested figure, which a renderer must show and index.
    """

    identifier: str
    kind: FigureKind
    title: str
    x_label: str
    y_label: str
    population: str
    series: tuple[FigureSeries, ...] = ()
    matrix: tuple[tuple[float | None, ...], ...] = ()
    matrix_labels: tuple[str, ...] = ()
    matrix_support: MetricSupport | None = None
    reason: str | None = None


def _identifier(value: str) -> str:
    """Use a safe stable filename token without erasing the original display label."""
    stem = re.sub("[^a-z0-9]+", "_", value.lower()).strip("_")[:40] or "value"
    return stem + "_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def _ecdf(
    name: str,
    values: tuple[float | int, ...],
    unit: SupportUnit,
) -> FigureSeries:
    """Freeze the exact right-continuous empirical CDF, with complete tie grouping."""
    counts = Counter(values)
    ordered = sorted(counts)
    cumulative = 0
    y: list[float] = []
    for value in ordered:
        cumulative += counts[value]
        y.append(cumulative / len(values))
    return FigureSeries(name, tuple(ordered), tuple(y), MetricSupport(unit=unit, n=len(values)))


def _distributions(
    identifier: str,
    title: str,
    x_label: str,
    values: tuple[tuple[str, tuple[float | int, ...]], ...],
    *,
    unit: SupportUnit = "IMAGE",
    population: str = "canonical tile/GSD variants",
) -> DatasetFigure:
    """Assemble exact cohort ECDFs; no histogram estimation, filtering or sampling."""
    available = any(cohort for _, cohort in values)
    series = tuple(_ecdf(name, cohort, unit) for name, cohort in values) if available else ()
    return DatasetFigure(
        identifier,
        "ECDF",
        title,
        x_label,
        "Empirical cumulative fraction",
        population,
        series=series,
        reason=None if series else "No eligible recorded values",
    )


def _coverage_chart(
    measured: DatasetMeasurement,
    field: str,
    *,
    population: str = "canonical_variant",
    title: str,
    identifier: str,
) -> DatasetFigure:
    """Pass categorical counts through captured coverage; missing values keep their own bar."""
    records = tuple(
        row
        for row in measured.coverage
        if row.population == population and row.field == field and row.split is not None
    )
    categories = sorted({row.value for row in records}, key=lambda v: (v is None, v or ""))
    labels = tuple(
        str(value) + " [recorded]"
        if value is not None
        else "Unbinned"
        if field == "bin_id"
        else "Unrecorded"
        for value in categories
    )
    series: list[FigureSeries] = []
    for split in ("train", "val", "test"):
        rows = tuple(row for row in records if row.split == split)
        if not rows:
            continue
        counts = {row.value: row.n for row in rows}
        series.append(
            FigureSeries(
                split,
                labels,
                tuple(counts.get(value, 0) for value in categories),
                MetricSupport(unit="IMAGE", n=rows[0].total),
            )
        )
    return DatasetFigure(
        identifier,
        "BAR",
        title,
        field,
        "Count",
        population,
        series=tuple(series),
        reason=None if series else "No eligible coverage records",
    )


def _split_samples(
    measured: DatasetMeasurement,
) -> tuple[tuple[Split, tuple[DatasetSample, ...]], ...]:
    """Partition only frozen canonical variants, never stored augmentation rows."""
    splits: tuple[Split, ...] = ("train", "val", "test")
    return tuple(
        (split, tuple(sample for sample in measured.samples if sample.key.split == split))
        for split in splits
    )


def dataset_figure_data(measured: DatasetMeasurement) -> tuple[DatasetFigure, ...]:
    """Freeze the complete supported dataset-chart inventory without source I/O.

    Counts retain null original-observation totals. Geometry ECDFs explicitly
    separate nonempty-truth masks from all explicit masks and components.
    Date/condition figures use recorded categories only; absence is unavailable.
    Related GSD variants remain visible, not silently treated as independent.
    """
    n = len(measured.samples)
    metrics = {metric.name: metric for metric in measured.metrics}
    count_names = (
        "stored_rows",
        "canonical_task_variants",
        "canonical_variants",
        "recorded_observations",
        "total_observations",
        "split_groups",
    )
    figures: list[DatasetFigure] = [
        DatasetFigure(
            "dataset_counts",
            "BAR",
            "Dataset population counts",
            "Population",
            "Count",
            "distinct stored, canonical, recorded-observation and group populations",
            series=(
                FigureSeries(
                    "dataset",
                    count_names,
                    tuple(metrics[name].value for name in count_names),
                    MetricSupport(unit="IMAGE", n=n),
                ),
            ),
        )
    ]
    for field, title in (
        ("label", "Canonical class composition"),
        ("source_annotation_state", "Recorded source annotation states"),
        ("prepared_mask_state", "Prepared source mask states"),
        ("mask_category", "Explicit mask and label populations"),
        ("shape", "Spatial shape coverage"),
        ("bin_id", "Recorded GSD-bin coverage"),
        ("gsd_provenance", "Nominal versus recorded non-nominal GSD"),
        ("observation_id", "Authoritative observation-ID coverage"),
        ("timestamp", "Acquisition-timestamp coverage"),
    ):
        figures.append(
            _coverage_chart(measured, field, title=title, identifier=field + "_coverage")
        )
    cohorts = _split_samples(measured)
    figures.append(
        DatasetFigure(
            "class_prevalence",
            "BAR",
            "Canonical positive-label prevalence",
            "Split",
            "Positive-label fraction",
            "canonical tile/GSD variants",
            series=tuple(
                FigureSeries(
                    split,
                    (split,),
                    (sum(sample.label for sample in samples) / len(samples) if samples else None,),
                    MetricSupport(unit="IMAGE", n=len(samples)),
                )
                for split, samples in cohorts
            ),
        )
    )
    for identifier, title, x_label, values in (
        (
            "gsd_lateral_ecdf",
            "Lateral GSD coverage",
            "Lateral GSD (m)",
            tuple((split, tuple(s.gsd_m[0] for s in samples)) for split, samples in cohorts),
        ),
        (
            "gsd_along_ecdf",
            "Along-track GSD coverage",
            "Along-track GSD (m)",
            tuple((split, tuple(s.gsd_m[1] for s in samples)) for split, samples in cohorts),
        ),
        (
            "gsd_anisotropy_ecdf",
            "GSD anisotropy",
            "max(GSD) / min(GSD)",
            tuple((split, tuple(s.gsd_anisotropy for s in samples)) for split, samples in cohorts),
        ),
        (
            "tile_area_m2_ecdf",
            "Local-GSD tile area",
            "Local-GSD tile area (m2)",
            tuple((split, tuple(s.tile_area_m2 for s in samples)) for split, samples in cohorts),
        ),
        (
            "positive_mask_area_px_ecdf",
            "Nonempty explicit-mask area",
            "Foreground area (pixel)",
            tuple(
                (
                    split,
                    tuple(
                        s.mask.area_px for s in samples if s.mask is not None and s.mask.area_px > 0
                    ),
                )
                for split, samples in cohorts
            ),
        ),
        (
            "positive_mask_area_m2_ecdf",
            "Nonempty explicit-mask local-GSD area",
            "Foreground area (m2)",
            tuple(
                (
                    split,
                    tuple(
                        s.mask.area_m2 for s in samples if s.mask is not None and s.mask.area_px > 0
                    ),
                )
                for split, samples in cohorts
            ),
        ),
        (
            "positive_mask_fraction_ecdf",
            "Nonempty explicit-mask area fraction",
            "Foreground fraction",
            tuple(
                (
                    split,
                    tuple(
                        s.mask.area_fraction
                        for s in samples
                        if s.mask is not None and s.mask.area_px > 0
                    ),
                )
                for split, samples in cohorts
            ),
        ),
        (
            "explicit_mask_components_ecdf",
            "Components per explicit mask",
            "Four-connected component count",
            tuple(
                (split, tuple(s.mask.n_components for s in samples if s.mask is not None))
                for split, samples in cohorts
            ),
        ),
    ):
        figures.append(_distributions(identifier, title, x_label, values))
    figures.append(
        DatasetFigure(
            "mask_border_touching",
            "BAR",
            "Border contact among explicit masks",
            "Split",
            "Border-touching mask fraction",
            "all explicit masks including empty masks",
            series=tuple(
                FigureSeries(
                    split,
                    (split,),
                    (
                        sum(s.mask.border_touching for s in samples if s.mask is not None) / count
                        if count
                        else None,
                    ),
                    MetricSupport(unit="IMAGE", n=count),
                )
                for split, samples in cohorts
                for count in (sum(s.mask is not None for s in samples),)
            ),
            reason=None
            if any(s.mask is not None for s in measured.samples)
            else "No explicit masks stored",
        )
    )
    for identifier, title, label, component_values in (
        (
            "component_area_px_ecdf",
            "Unfiltered ground-truth component area",
            "Component area (pixel)",
            tuple(component.area_px for component in measured.components),
        ),
        (
            "component_area_m2_ecdf",
            "Unfiltered local-GSD component area",
            "Component area (m2)",
            tuple(component.area_m2 for component in measured.components),
        ),
    ):
        figures.append(
            _distributions(
                identifier,
                title,
                label,
                (("all components", component_values),),
                unit="COMPONENT",
                population="unfiltered four-connected explicit-mask components,"
                " not physical plumes",
            )
        )
    pixels = next((cohort.pixels for cohort in measured.pixels if cohort.split is None), None)
    if pixels is not None:
        for band in pixels.bands:
            figures.append(
                DatasetFigure(
                    "pixel_histogram_" + _identifier(band.name),
                    "HISTOGRAM",
                    band.name + " processed-unit pixel distribution",
                    "Processed unit value",
                    "Pixel count",
                    "all pixels in canonical variants, unioned across tasks",
                    series=(
                        FigureSeries(
                            band.name,
                            pixels.histogram_bin_edges,
                            band.histogram_counts,
                            MetricSupport(unit="PIXEL", n=pixels.n_pixels),
                        ),
                    ),
                )
            )
        figures.append(
            DatasetFigure(
                "pixel_moments",
                "BAR",
                "Processed-unit pixel population moments",
                "Band",
                "Processed unit value",
                "all pixels in canonical variants, unioned across tasks",
                series=tuple(
                    FigureSeries(
                        name,
                        tuple(band.name for band in pixels.bands),
                        values,
                        MetricSupport(unit="PIXEL", n=pixels.n_pixels),
                    )
                    for name, values in (
                        ("mean", tuple(band.mean for band in pixels.bands)),
                        ("population std", tuple(band.std for band in pixels.bands)),
                        ("minimum", tuple(band.minimum for band in pixels.bands)),
                        ("maximum", tuple(band.maximum for band in pixels.bands)),
                    )
                ),
            )
        )
        figures.append(
            DatasetFigure(
                "pixel_endpoints",
                "BAR",
                "Processed-unit pixel endpoints",
                "Band",
                "Pixel count",
                "processed endpoints, not raw sensor saturation",
                series=tuple(
                    FigureSeries(
                        endpoint,
                        tuple(band.name for band in pixels.bands),
                        values,
                        MetricSupport(unit="PIXEL", n=pixels.n_pixels),
                    )
                    for endpoint, values in (
                        ("at zero", tuple(band.at_zero for band in pixels.bands)),
                        ("at one", tuple(band.at_one for band in pixels.bands)),
                    )
                ),
            )
        )
        figures.append(
            DatasetFigure(
                "pixel_correlations",
                "MATRIX",
                "Processed-pixel band correlations",
                "Band",
                "Band",
                "all canonical-variant pixels; constant-band pairs unavailable",
                matrix=pixels.correlations,
                matrix_labels=measured.identity.band_names,
                matrix_support=MetricSupport(unit="PIXEL", n=pixels.n_pixels),
            )
        )
    for population in ("canonical_variant", "recorded_observation"):
        figure = _coverage_chart(
            measured,
            "month_utc",
            population=population,
            title="Recorded acquisition month (" + population + ")",
            identifier="timestamps_" + population,
        )
        has_dates = any(
            row.population == population and row.field == "month_utc" and row.value is not None
            for row in measured.coverage
        )
        figures.append(
            figure
            if has_dates
            else DatasetFigure(
                figure.identifier,
                figure.kind,
                figure.title,
                figure.x_label,
                figure.y_label,
                figure.population,
                reason="No acquisition timestamps recorded for this population",
            )
        )
    fields = sorted({row.field for row in measured.coverage if row.field.startswith("condition:")})
    for name in fields:
        for population in ("canonical_variant", "recorded_observation"):
            figures.append(
                _coverage_chart(
                    measured,
                    name,
                    population=population,
                    title=name + " coverage (" + population + ")",
                    identifier=_identifier(name) + "_" + population,
                )
            )
    if not fields:
        figures.append(
            DatasetFigure(
                "conditions_unavailable",
                "BAR",
                "Recorded condition coverage",
                "Condition",
                "Count",
                "recorded categorical conditions",
                reason="No categorical conditions were recorded",
            )
        )
    return tuple(figures)
