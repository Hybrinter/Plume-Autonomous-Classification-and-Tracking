"""Frozen dataset figure-coordinate references."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
from flight.libs.types import Ok
from tools.ml_models.analysis.dataset import measure_dataset
from tools.ml_models.analysis.dataset_figures import _distributions, dataset_figure_data


def test_counts_and_pixels_match_frozen_measurement(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    measured = measure_dataset(build_synthetic_dataset(tmp_path / "ds"))
    assert isinstance(measured, Ok)
    frozen = measured.value
    figures = {figure.identifier: figure for figure in dataset_figure_data(frozen)}
    counts = figures["dataset_counts"].series[0]
    values = {metric.name: metric.value for metric in frozen.metrics}
    assert counts.y == tuple(values[cast(str, name)] for name in counts.x)
    assert values["total_observations"] is None
    pixels = next(cohort.pixels for cohort in frozen.pixels if cohort.split is None)
    histograms = [figure for figure in figures.values() if figure.kind == "HISTOGRAM"]
    assert len(histograms) == len(pixels.bands)
    for figure, band in zip(histograms, pixels.bands, strict=True):
        series = figure.series[0]
        assert series.x == pixels.histogram_bin_edges
        assert series.y == band.histogram_counts
        assert series.support.unit == "PIXEL"
        assert series.support.n == pixels.n_pixels
    correlation = figures["pixel_correlations"]
    assert correlation.matrix == pixels.correlations
    assert correlation.matrix_labels == frozen.identity.band_names
    assert correlation.matrix_support is not None
    assert correlation.matrix_support.n == pixels.n_pixels
    moments = {series.name: series for series in figures["pixel_moments"].series}
    assert moments["mean"].y == tuple(band.mean for band in pixels.bands)
    assert moments["population std"].y == tuple(band.std for band in pixels.bands)
    assert all(series.support.n == pixels.n_pixels for series in moments.values())


def test_gsd_ecdf_coordinates_are_exact_and_unaugmented(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    measured = measure_dataset(build_synthetic_dataset(tmp_path / "ds"))
    assert isinstance(measured, Ok)
    frozen = measured.value
    figures = {figure.identifier: figure for figure in dataset_figure_data(frozen)}
    for series in figures["gsd_lateral_ecdf"].series:
        samples = tuple(sample for sample in frozen.samples if sample.key.split == series.name)
        values = np.array([sample.gsd_m[0] for sample in samples])
        unique = np.unique(values)
        assert series.x == tuple(unique)
        assert series.y == tuple(
            float(np.count_nonzero(values <= edge) / len(values)) for edge in unique
        )
        assert series.support.n == len(samples)
    assert figures["timestamps_canonical_variant"].reason
    assert figures["conditions_unavailable"].reason
    for series in figures["class_prevalence"].series:
        samples = tuple(sample for sample in frozen.samples if sample.key.split == series.name)
        assert series.support.n == len(samples)
        assert series.y == (sum(sample.label for sample in samples) / len(samples),)


def test_empty_masks_cannot_inflate_positive_area_coordinates(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    measured = measure_dataset(build_synthetic_dataset(tmp_path / "ds"))
    assert isinstance(measured, Ok)
    frozen = measured.value
    samples = tuple(
        replace(sample, mask=replace(sample.mask, area_px=0, area_m2=0.0, area_fraction=0.0))
        if sample.mask is not None
        else sample
        for sample in frozen.samples
    )
    figures = {
        figure.identifier: figure
        for figure in dataset_figure_data(replace(frozen, samples=samples, components=()))
    }
    figure = figures["positive_mask_area_px_ecdf"]
    assert figure.reason
    assert not figure.series
    assert figures["component_area_px_ecdf"].reason


def test_no_masks_keeps_border_contact_unavailable(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    measured = measure_dataset(build_synthetic_dataset(tmp_path / "ds"))
    assert isinstance(measured, Ok)
    frozen = measured.value
    samples = tuple(replace(sample, mask=None, mask_key=None) for sample in frozen.samples)
    figures = {
        figure.identifier: figure
        for figure in dataset_figure_data(replace(frozen, samples=samples, components=()))
    }
    border = figures["mask_border_touching"]
    assert border.reason
    assert all(series.y == (None,) and series.support.n == 0 for series in border.series)


def test_empty_split_support_cannot_disappear_from_a_supported_distribution() -> None:
    figure = _distributions(
        "reference",
        "Reference",
        "Value",
        (("train", (1, 1, 2)), ("val", ()), ("test", (3,))),
    )
    assert figure.reason is None
    series = {row.name: row for row in figure.series}
    assert set(series) == {"train", "val", "test"}
    assert series["val"].support.n == 0
    assert series["val"].x == () and series["val"].y == ()
    assert series["train"].x == (1, 2)
    assert series["train"].y == (2 / 3, 1)
