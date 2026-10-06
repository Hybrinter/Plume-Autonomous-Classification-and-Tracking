"""Segmentor and generalization wrappers share the frozen model renderer."""

from __future__ import annotations

import numpy as np
from flight.libs.types import Ok
from matplotlib.figure import Figure
from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.contracts import MetricSupport, Task
from tools.ml_models.analysis.model_figures import (
    FigureIdentity,
    ModelFigure,
    ModelPoint,
    ModelSeries,
)
from tools.ml_models.analysis.plots.generalization import render_generalization_figures
from tools.ml_models.analysis.plots.segmentation import render_segmentation_figures

_SUPPORT = MetricSupport(unit="IMAGE", n=4)


def _figure(task: Task, **overrides: object) -> ModelFigure:
    values: dict[str, object] = {
        "identifier": "extent_success",
        "identity": FigureIdentity(
            dataset_hash="a" * 64,
            dataset_manifest_hash="b" * 64,
            checkpoint_hash="c" * 64,
            task=task,
            split="val",
        ),
        "title": "Captured extent success",
        "x_label": "centroid error (px)",
        "y_label": "fraction of truth components",
        "population": "val image cohort",
    }
    values.update(overrides)
    return ModelFigure(**values)  # type: ignore[arg-type]


def _result_figure(record: ModelFigure) -> Figure:
    from tools.ml_models.analysis.plots.model import model_figure

    result = model_figure(record, PlotConfig())
    assert isinstance(result, Ok)
    return result.value


def test_segmentor_family_paths_and_names() -> None:
    record = _figure(
        "segmentor",
        series=(ModelSeries("s", (0.0, 1.0), (0.5, 1.0), _SUPPORT),),
    )
    result = render_segmentation_figures((record,), PlotConfig(formats=("png", "svg"), dpi=72))
    assert isinstance(result, Ok)
    assert {file.path for file in result.value.files} == {
        "figures/segmentor/val/extent_success.png",
        "figures/segmentor/val/extent_success.svg",
    }
    assert result.value.outputs[0].name == "model_figure:segmentor:val:extent_success"


def test_generalization_family_paths_and_names() -> None:
    record = _figure(
        "segmentor",
        identifier="stratum_metric",
        series=(ModelSeries("s", (0.0, 1.0), (0.5, 1.0), _SUPPORT),),
    )
    result = render_generalization_figures((record,), PlotConfig(dpi=72, formats=("png",)))
    assert isinstance(result, Ok)
    assert {file.path for file in result.value.files} == {
        "figures/generalization/val/stratum_metric.png"
    }
    output = result.value.outputs[0]
    assert output.name == "model_figure:generalization:val:stratum_metric"


def test_miss_inclusive_curve_keeps_distinct_coordinates() -> None:
    record = _figure(
        "segmentor",
        identifier="centroid_success",
        series=(
            ModelSeries(
                "miss-inclusive success",
                (0.0, 1.0, 2.0),
                (0.5, 0.75, 1.0),
                _SUPPORT,
            ),
            ModelSeries(
                "conditional matched ECDF",
                (0.0, 1.0, 2.0),
                (0.0, 0.5, 1.0),
                _SUPPORT,
            ),
        ),
    )
    figure = _result_figure(record)
    import matplotlib.pyplot as plt

    try:
        labelled = {
            str(line.get_label()).split(" (")[0]: line
            for line in figure.axes[0].lines
            if line.get_label() not in ("", "_nolegend_")
        }
        np.testing.assert_allclose(
            np.asarray(labelled["miss-inclusive success"].get_ydata(), dtype=float),
            (0.5, 0.75, 1.0),
        )
        np.testing.assert_allclose(
            np.asarray(labelled["conditional matched ECDF"].get_ydata(), dtype=float),
            (0.0, 0.5, 1.0),
        )
    finally:
        plt.close(figure)


def test_copied_interval_and_operating_point_render_verbatim() -> None:
    record = _figure(
        "segmentor",
        identifier="stratum_iou",
        series=(
            ModelSeries(
                "approximate grouped IoU",
                (0.0, 1.0, 2.0),
                (0.4, 0.6, 0.8),
                _SUPPORT,
                lower=(0.1, 0.5, 0.9),
                upper=(0.6, 0.7, 0.95),
            ),
        ),
        points=(ModelPoint("captured operating point", 1.0, 0.6),),
        notes=("approximate histogram over frozen bin edges",),
    )
    figure = _result_figure(record)
    import matplotlib.pyplot as plt

    try:
        axes = figure.axes[0]
        stars = [line for line in axes.lines if line.get_marker() == "*"]
        assert len(stars) == 1
        np.testing.assert_allclose(np.asarray(stars[0].get_ydata(), dtype=float), [0.6])
        from matplotlib.collections import LineCollection

        segments = [
            segment
            for collection in axes.collections
            if isinstance(collection, LineCollection)
            for segment in collection.get_segments()
        ]
        assert len(segments) == 3
        endpoints = sorted((float(segment[0][1]), float(segment[1][1])) for segment in segments)
        assert endpoints == [(0.1, 0.6), (0.5, 0.7), (0.9, 0.95)]
        footer = "\n".join(text.get_text() for text in figure.texts)
        assert "approximate histogram" in footer
        assert "val image cohort" in footer
    finally:
        plt.close(figure)


def test_generalization_matrix_nulls_and_shared_range() -> None:
    record = _figure(
        "segmentor",
        identifier="cohort_heatmap",
        kind="MATRIX",
        x_categories=("site-a", "site-b"),
        y_categories=("low-gsd", "high-gsd"),
        matrix=((0.5, None), (0.7, 0.9)),
        matrix_support=((4, 0), (4, 4)),
        matrix_range=(0.0, 1.0),
        notes=("cells share the frozen full-matrix range",),
    )
    result = render_generalization_figures((record,), PlotConfig(dpi=72, formats=("png",)))
    assert isinstance(result, Ok)
    rendered = _result_figure(record)
    import matplotlib.pyplot as plt

    try:
        image = rendered.axes[0].images[0]
        assert image.get_clim() == (0.0, 1.0)
        raw = image.get_array()
        assert raw is not None
        data = np.ma.masked_invalid(np.asarray(raw, dtype=float))
        assert bool(np.ma.getmaskarray(data)[0, 1])
        assert data[1, 1] == 0.9
    finally:
        plt.close(rendered)
