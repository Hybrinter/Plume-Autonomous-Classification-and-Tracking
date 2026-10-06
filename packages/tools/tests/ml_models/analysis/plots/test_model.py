"""Frozen model-figure rendering checks: verbatim coordinates, styles, matrices."""

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from flight.libs.types import Err, Ok
from matplotlib.figure import Figure
from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.contracts import MetricSupport, Split
from tools.ml_models.analysis.model_figures import (
    DrawStyle,
    FigureIdentity,
    ModelFigure,
    ModelPoint,
    ModelSeries,
)
from tools.ml_models.analysis.plots.classifier import render_classifier_figures
from tools.ml_models.analysis.plots.model import model_figure, render_model_figures

_IDENTITY = FigureIdentity(
    dataset_hash="a" * 64,
    dataset_manifest_hash="b" * 64,
    checkpoint_hash="c" * 64,
    task="classifier",
    split="val",
)
_SUPPORT = MetricSupport(unit="IMAGE", n=4)


def _figure(**overrides: object) -> ModelFigure:
    values: dict[str, object] = {
        "identifier": "precision_recall",
        "identity": _IDENTITY,
        "title": "Captured precision recall",
        "x_label": "recall (fraction)",
        "y_label": "precision (fraction)",
        "population": "val image cohort",
    }
    values.update(overrides)
    return ModelFigure(**values)  # type: ignore[arg-type]


def _result_figure(record: ModelFigure) -> Figure:
    result = model_figure(record, PlotConfig())
    assert isinstance(result, Ok)
    return result.value


def test_series_styles_and_exact_coordinates() -> None:
    record = _figure(
        series=(
            ModelSeries("precision_recall", (0.0, 0.5, 1.0), (1.0, 0.5, 0.5), _SUPPORT, "PRE"),
            ModelSeries("roc", (0.0, 0.5, 1.0), (0.0, 0.4, 1.0), _SUPPORT, "LINE"),
            ModelSeries("gain", (0.0, 1.0), (0.2, 0.9), _SUPPORT, "POST"),
        ),
    )
    figure = _result_figure(record)
    lines = [line for line in figure.axes[0].lines if line.get_label() not in ("", "_nolegend_")]
    draw = {str(line.get_label()).split(" (")[0]: line for line in lines}
    assert draw["precision_recall"].get_drawstyle() == "steps-pre"
    np.testing.assert_allclose(
        np.asarray(draw["precision_recall"].get_xdata(), dtype=float), (0.0, 0.5, 1.0)
    )
    np.testing.assert_allclose(
        np.asarray(draw["precision_recall"].get_ydata(), dtype=float), (1.0, 0.5, 0.5)
    )
    assert draw["roc"].get_drawstyle() == "default"
    np.testing.assert_allclose(np.asarray(draw["roc"].get_ydata(), dtype=float), (0.0, 0.4, 1.0))
    assert draw["gain"].get_drawstyle() == "steps-post"
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_recorded_operating_point_off_grid_is_drawn_exactly() -> None:
    record = _figure(
        series=(ModelSeries("threshold_precision", (0.0, 0.5, 1.0), (0.5, 1.0, 0.0), _SUPPORT),),
        points=(ModelPoint("Captured operating threshold: 0.3", 0.3, 0.55),),
    )
    figure = _result_figure(record)
    markers = [line for line in figure.axes[0].lines if line.get_marker() == "*"]
    assert len(markers) == 1
    np.testing.assert_allclose(np.asarray(markers[0].get_xdata(), dtype=float), [0.3])
    np.testing.assert_allclose(np.asarray(markers[0].get_ydata(), dtype=float), [0.55])
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_singleton_series_and_null_gaps_visible() -> None:
    record = _figure(
        series=(
            ModelSeries("gapped", (0.0, 0.5, 1.0), (1.0, None, 0.5), _SUPPORT, "LINE"),
            ModelSeries("single", (0.5,), (0.25,), _SUPPORT, "POINT"),
        ),
    )
    figure = _result_figure(record)
    gapped = next(
        line for line in figure.axes[0].lines if str(line.get_label()).startswith("gapped")
    )
    assert np.isnan(np.asarray(gapped.get_ydata(), dtype=float)[1])
    assert gapped.get_marker() == "o"
    single = next(
        line for line in figure.axes[0].lines if str(line.get_label()).startswith("single")
    )
    assert single.get_linestyle() == "None"
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_interval_endpoints_are_absolute_not_centered() -> None:
    record = _figure(
        series=(
            ModelSeries(
                "accuracy",
                (0.0,),
                (0.5,),
                _SUPPORT,
                "POINT",
                lower=(0.1,),
                upper=(0.9,),
            ),
        ),
        x_categories=("val",),
    )
    figure = _result_figure(record)
    from matplotlib.collections import LineCollection

    collections = [c for c in figure.axes[0].collections if isinstance(c, LineCollection)]
    assert len(collections) == 1
    segments = collections[0].get_segments()
    np.testing.assert_allclose(segments[0][0], (0.0, 0.1))
    np.testing.assert_allclose(segments[0][-1], (0.0, 0.9))
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_bar_uses_categories_heights_and_point_support() -> None:
    record = _figure(
        identifier="class_counts",
        x_categories=("negative", "positive"),
        series=(
            ModelSeries(
                "images",
                (0.0, 1.0),
                (2.0, 3.0),
                _SUPPORT,
                "BAR",
                point_support=(2, 3),
            ),
        ),
    )
    figure = _result_figure(record)
    from matplotlib.patches import Rectangle

    bars = [
        patch.get_height()
        for patch in figure.axes[0].patches
        if isinstance(patch, Rectangle) and patch.get_height() > 0
    ]
    assert bars == [2.0, 3.0]
    labels = [tick.get_text() for tick in figure.axes[0].get_xticklabels()]
    assert labels == ["negative", "positive"]
    texts = [text.get_text() for text in figure.axes[0].texts]
    assert "n=2" in texts and "n=3" in texts
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_matrix_null_cells_masked_with_counts_annotations() -> None:
    record = _figure(
        identifier="confusion_truth_normalized",
        kind="MATRIX",
        x_label="Predicted class",
        y_label="Truth class",
        x_categories=("negative", "positive"),
        y_categories=("negative", "positive"),
        matrix=((0.5, 0.5), (None, None)),
        matrix_support=((1, 1), (0, 0)),
        matrix_range=(0.0, 1.0),
    )
    figure = _result_figure(record)
    axes = figure.axes[0]
    assert axes.get_xlabel() == "Predicted class"
    assert axes.get_ylabel() == "Truth class"
    image = axes.images[0]
    assert image.get_clim() == (0.0, 1.0)
    raw = image.get_array()
    assert raw is not None
    array = np.ma.filled(raw, np.nan)
    assert np.isnan(array[1, 0]) and np.isnan(array[1, 1])
    assert array[0, 0] == 0.5
    texts = {text.get_text() for text in axes.texts}
    assert "n/a" in texts
    assert any(text.startswith("n=1") or "n=1" in text for text in texts)
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_matrix_data_bounds_used_when_range_absent() -> None:
    record = _figure(
        identifier="confusion_counts",
        kind="MATRIX",
        x_categories=("negative", "positive"),
        y_categories=("negative", "positive"),
        matrix=((4.0, 1.0), (2.0, 9.0)),
        matrix_support=((4, 1), (2, 9)),
    )
    figure = _result_figure(record)
    assert figure.axes[0].images[0].get_clim() == (1.0, 9.0)
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_unavailable_reason_and_namespaced_output() -> None:
    record = _figure(reason="No eligible captured values")
    result = render_classifier_figures((record,), PlotConfig())
    assert isinstance(result, Ok)
    output = result.value.outputs[0]
    assert output.name == "model_figure:classifier:val:precision_recall"
    assert output.status == "UNAVAILABLE" and output.reason == "No eligible captured values"
    assert any(
        file.path == "figures/classifier/val/precision_recall.png" for file in result.value.files
    )


def test_render_files_are_namespaced_per_family_and_split() -> None:
    record = _figure(series=(ModelSeries("s", (0.0, 1.0), (0.0, 1.0), _SUPPORT),))
    cfg = PlotConfig(formats=("png", "svg"), dpi=100)
    result = render_classifier_figures((record,), cfg)
    assert isinstance(result, Ok)
    paths = {file.path for file in result.value.files}
    assert paths == {
        "figures/classifier/val/precision_recall.png",
        "figures/classifier/val/precision_recall.svg",
    }
    assert result.value.outputs[0].status == "AVAILABLE"


@pytest.mark.parametrize(
    "overrides",
    (
        {"series": (ModelSeries("bad", (0.0,), (1.0, 2.0), _SUPPORT),)},
        {"series": (ModelSeries("bad", (0.0, float("nan")), (1.0, 2.0), _SUPPORT),)},
        {
            "series": (
                ModelSeries(
                    "bad", (0.0, 1.0), (1.0, 2.0), _SUPPORT, lower=(0.0,), upper=(0.0, 0.0)
                ),
            )
        },
        {"series": (ModelSeries("bad", (0.0,), (1.0,), _SUPPORT, point_support=(1, 2)),)},
        {"points": (ModelPoint("p", float("inf"), 0.5),)},
        {
            "kind": "MATRIX",
            "x_categories": ("a", "b"),
            "y_categories": ("a",),
            "matrix": ((1.0, 2.0),),
            "matrix_support": ((1,),),
        },
        {
            "kind": "MATRIX",
            "x_categories": ("a",),
            "y_categories": ("a",),
            "matrix": ((1.0,),),
            "matrix_support": ((1,),),
            "matrix_range": (2.0, 1.0),
        },
    ),
)
def test_malformed_shapes_rejected_without_truncation(overrides: dict[str, object]) -> None:
    assert isinstance(model_figure(_figure(**overrides), PlotConfig()), Err)


def test_unsafe_identifiers_family_and_duplicates_rejected() -> None:
    good = _figure(series=(ModelSeries("s", (0.0,), (1.0,), _SUPPORT),))
    assert isinstance(model_figure(_figure(identifier="a/b"), PlotConfig()), Err)
    assert isinstance(render_model_figures((good,), PlotConfig(), family="../escape"), Err)
    other = FigureIdentity(
        dataset_hash="a" * 64,
        dataset_manifest_hash=None,
        checkpoint_hash=None,
        task="classifier",
        split=cast("Split", "../bad"),
    )
    assert isinstance(model_figure(_figure(identity=other), PlotConfig()), Err)
    assert isinstance(render_classifier_figures((good, good), PlotConfig()), Err)


def test_same_identifier_across_splits_is_unique_per_split() -> None:
    train = _figure(
        identity=replace(_IDENTITY, split="train"),
        series=(ModelSeries("s", (0.0, 1.0), (0.0, 1.0), _SUPPORT),),
    )
    val = _figure(series=(ModelSeries("s", (0.0, 1.0), (0.0, 1.0), _SUPPORT),))
    result = render_classifier_figures((train, val), PlotConfig(formats=("png",)))
    assert isinstance(result, Ok)
    assert {file.path for file in result.value.files} == {
        "figures/classifier/train/precision_recall.png",
        "figures/classifier/val/precision_recall.png",
    }
    assert isinstance(render_classifier_figures((val, val), PlotConfig()), Err)


def test_bar_keeps_frozen_x_null_height_and_raw_zero() -> None:
    record = _figure(
        identifier="class_counts",
        x_categories=("negative", "positive"),
        series=(ModelSeries("images", (2.0, 5.0), (0.0, None), _SUPPORT, "BAR"),),
    )
    figure = _result_figure(record)
    axes = figure.axes[0]
    from matplotlib.patches import Rectangle

    bars = [patch for patch in axes.patches if isinstance(patch, Rectangle)]
    positions = sorted(patch.get_x() + patch.get_width() / 2.0 for patch in bars)
    np.testing.assert_allclose(positions, (2.0, 5.0))
    heights = {
        round(patch.get_x() + patch.get_width() / 2.0, 6): patch.get_height() for patch in bars
    }
    assert heights[2.0] == 0.0
    assert np.isnan(heights[5.0])
    ticks = [
        tick
        for tick, label in zip(axes.get_xticks(), axes.get_xticklabels(), strict=True)
        if label.get_text()
    ]
    np.testing.assert_allclose(ticks, (2.0, 5.0))
    texts = {text.get_text() for text in axes.texts}
    assert "n/a" in texts
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_interval_need_not_enclose_estimate() -> None:
    record = _figure(
        series=(
            ModelSeries(
                "accuracy",
                (0.0, 1.0),
                (0.3, 0.5),
                _SUPPORT,
                "POINT",
                lower=(0.4, 0.1),
                upper=(0.6, 0.9),
            ),
        ),
    )
    figure = _result_figure(record)
    from matplotlib.collections import LineCollection

    collections = [item for item in figure.axes[0].collections if isinstance(item, LineCollection)]
    assert len(collections) == 1
    segments = sorted(float(item[0][1]) for item in collections[0].get_segments())
    np.testing.assert_allclose(segments, (0.1, 0.4))
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_baseline_style_is_limited_to_baseline_names() -> None:
    baseline = _figure(
        series=(ModelSeries("chance diagonal", (0.0, 1.0), (0.0, 1.0), _SUPPORT),),
    )
    figure = _result_figure(baseline)
    line = next(item for item in figure.axes[0].lines if str(item.get_label()).startswith("chance"))
    assert line.get_linestyle() == "--"
    assert line.get_zorder() == 1
    import matplotlib.pyplot as plt

    plt.close(figure)
    curve = _figure(
        series=(ModelSeries("calibration_reliability", (0.0, 1.0), (0.0, 1.0), _SUPPORT),),
    )
    figure = _result_figure(curve)
    line = next(
        item
        for item in figure.axes[0].lines
        if str(item.get_label()).startswith("calibration_reliability")
    )
    assert line.get_linestyle() == "-"
    assert line.get_zorder() == 2
    plt.close(figure)


def test_matrix_dark_cells_use_high_contrast_text() -> None:
    record = _figure(
        identifier="confusion_counts",
        kind="MATRIX",
        x_categories=("negative", "positive"),
        y_categories=("negative", "positive"),
        matrix=((900.0, 1.0), (2.0, 4.0)),
        matrix_support=((900, 1), (2, 4)),
    )
    figure = _result_figure(record)
    colors = {str(text.get_color()).upper() for text in figure.axes[0].texts}
    assert colors == {"#111111", "#FFFFFF"}
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_population_and_notes_footer_for_matrix_and_placeholder() -> None:
    matrix = _figure(
        identifier="confusion_counts",
        kind="MATRIX",
        population="val image cohort",
        notes=("Row-normalized captured counts",),
        x_categories=("negative", "positive"),
        y_categories=("negative", "positive"),
        matrix=((1.0, 0.0), (0.0, 1.0)),
        matrix_support=((1, 0), (0, 1)),
    )
    figure = _result_figure(matrix)
    joined = "\n".join(text.get_text() for text in figure.texts)
    assert "val image cohort" in joined
    assert "Row-normalized captured counts" in joined
    import matplotlib.pyplot as plt

    plt.close(figure)
    placeholder = _result_figure(
        _figure(reason="No eligible captured values", notes=("frozen note",))
    )
    joined = "\n".join(text.get_text() for text in placeholder.texts)
    assert "val image cohort" in joined
    assert "frozen note" in joined
    plt.close(placeholder)


@pytest.mark.parametrize(
    "overrides",
    (
        {"kind": "LINE"},
        {"series": (ModelSeries("bad", (0.0,), (1.0,), _SUPPORT, cast("DrawStyle", "SPLINE")),)},
        {"series": (ModelSeries("bad", (0.0,), (1.0,), _SUPPORT, lower=(0.1,), upper=(None,)),)},
        {"series": (ModelSeries("bad", (0.0,), (1.0,), _SUPPORT, lower=(0.9,), upper=(0.1,)),)},
        {
            "series": (
                ModelSeries(
                    "bad",
                    (0.0,),
                    (1.0,),
                    _SUPPORT,
                    lower=(float("inf"),),
                    upper=(2.0,),
                ),
            )
        },
        {"series": (ModelSeries("bad", (0.0,), (1.0,), _SUPPORT, point_support=(True,)),)},
        {"series": (ModelSeries("bad", (0.0,), (1.0,), _SUPPORT, point_support=(-1,)),)},
        {
            "kind": "MATRIX",
            "x_categories": ("a",),
            "y_categories": ("a",),
            "matrix": ((1.0,),),
            "matrix_support": ((-1,),),
        },
        {
            "kind": "MATRIX",
            "x_categories": ("a",),
            "y_categories": ("a",),
            "matrix": ((1.0,),),
            "matrix_support": ((True,),),
        },
        {
            "kind": "MATRIX",
            "x_categories": ("a",),
            "y_categories": ("a",),
            "matrix": ((1.0,),),
            "matrix_support": ((1, 2),),
        },
        {
            "x_categories": ("a", "b"),
            "series": (ModelSeries("bad", (0.0,), (1.0,), _SUPPORT),),
        },
    ),
)
def test_invalid_records_return_err(overrides: dict[str, object]) -> None:
    assert isinstance(model_figure(_figure(**overrides), PlotConfig()), Err)


def test_figures_are_closed() -> None:
    import matplotlib.pyplot as plt

    record = _figure(series=(ModelSeries("s", (0.0,), (1.0,), _SUPPORT),))
    assert isinstance(render_classifier_figures((record,), PlotConfig()), Ok)
    assert plt.get_fignums() == []
    assert isinstance(render_classifier_figures((_figure(identifier="a/b"),), PlotConfig()), Err)
    assert plt.get_fignums() == []


def test_renderer_never_recomputes_or_reads_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    def _bomb(*args: object, **kwargs: object) -> None:
        raise AssertionError("renderer touched selection/source helpers")

    import tools.ml_models.analysis.prediction_selections as selections

    monkeypatch.setattr(selections, "prediction_gallery_data", _bomb)
    record = _figure(series=(ModelSeries("s", (0.0,), (1.0,), _SUPPORT),))
    assert isinstance(render_classifier_figures((record,), PlotConfig()), Ok)
