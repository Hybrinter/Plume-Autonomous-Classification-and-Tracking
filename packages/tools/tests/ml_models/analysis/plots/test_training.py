"""Tests for frozen training-history figure rendering."""

import matplotlib
import numpy as np
import pytest
from flight.libs.types import Err, Ok
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.plots.training import (
    render_training_figures,
    training_figure,
)
from tools.ml_models.analysis.training_figures import (
    AxisScale,
    TrainingFigure,
    TrainingMarker,
    TrainingSeries,
)

_CFG = PlotConfig(formats=("png", "svg"), dpi=72, width_inches=7.0, height_inches=4.5)


def _series(
    name: str,
    x: tuple[float, ...],
    y: tuple[float | None, ...],
    *,
    points_only: bool = False,
    unit: str = "IMAGE",
    n: int | None = None,
) -> TrainingSeries:
    return TrainingSeries(
        name,
        x,
        y,
        len(x) if n is None else n,
        "synthetic captured population",
        points_only,
        unit,
    )


def _record(
    series: tuple[TrainingSeries, ...],
    *,
    identifier: str = "loss",
    x_scale: AxisScale = "linear",
    y_scale: AxisScale = "linear",
    markers: tuple[TrainingMarker, ...] = (),
    run_status: str = "COMPLETED",
    reason: str | None = None,
    warnings: tuple[str, ...] = (),
) -> TrainingFigure:
    return TrainingFigure(
        identifier,
        "Captured objective",
        "Successful optimizer updates",
        "Configured weighted objective",
        series,
        markers,
        run_status,
        x_scale,
        y_scale,
        reason,
        warnings,
    )


def _built(record: TrainingFigure, cfg: PlotConfig = _CFG) -> Figure:
    built = training_figure(record, cfg)
    assert isinstance(built, Ok)
    return built.value


def _xy(line: Line2D) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(line.get_xdata(), dtype=np.float64),
        np.asarray(line.get_ydata(), dtype=np.float64),
    )


def _vline_positions(figure: Figure) -> list[float]:
    positions: list[float] = []
    for line in figure.axes[0].lines:
        xs, _ = _xy(line)
        if len(xs) == 2 and xs[0] == xs[1]:
            positions.append(float(xs[0]))
    return positions


def _legend_texts(figure: Figure) -> list[str]:
    legend = figure.axes[0].get_legend()
    if legend is None:
        return []
    return [text.get_text() for text in legend.get_texts()]


def _figure_text(figure: Figure) -> str:
    return " ".join(text.get_text() for text in figure.texts) + " ".join(
        text.get_text() for text in figure.axes[0].texts
    )


def test_log_scales_and_exact_coordinates_render_verbatim() -> None:
    record = _record(
        (_series("canonical train", (1.0, 2.0, 3.0), (3.0, 1.0, 2.0)),),
        x_scale="log",
        y_scale="log",
    )
    figure = _built(record)
    axes = figure.axes[0]
    assert axes.get_xscale() == "log" and axes.get_yscale() == "log"
    (line,) = axes.lines
    xs, ys = _xy(line)
    assert xs.tolist() == [1.0, 2.0, 3.0]
    assert ys.tolist() == [3.0, 1.0, 2.0]
    assert line.get_marker() == "o"
    figure.clf()


def test_semilog_companion_keeps_x_linear() -> None:
    record = _record(
        (_series("validation", (1.0, 2.0), (2.0, 2.5)),),
        x_scale="linear",
        y_scale="log",
    )
    figure = _built(record)
    axes = figure.axes[0]
    assert axes.get_xscale() == "linear" and axes.get_yscale() == "log"
    figure.clf()


def test_zero_and_null_mask_line_breaks_without_epsilon_floor() -> None:
    record = _record(
        (_series("canonical train", (0.0, 1.0, 2.0, 3.0), (3.0, 0.0, None, 1.0)),),
        x_scale="log",
        y_scale="log",
    )
    figure = _built(record)
    (line,) = figure.axes[0].lines
    xs, ys = _xy(line)
    assert len(xs) == 4
    assert np.isnan(xs[0]) and np.isnan(ys[0])
    assert np.isnan(ys[1]) and np.isnan(ys[2])
    assert ys[3] == 1.0
    assert not np.any(ys[~np.isnan(ys)] <= 0.0)
    labels = " ".join(_legend_texts(figure))
    assert "outside logarithmic domain" in labels
    assert "unavailable" in labels
    figure.clf()


def test_linear_companion_shows_zeros_and_stays_available() -> None:
    record = _record(
        (_series("canonical train", (1.0, 2.0), (0.0, 1.0)),),
    )
    rendered = render_training_figures((record,), _CFG)
    assert isinstance(rendered, Ok)
    assert rendered.value.outputs[0].status == "AVAILABLE"
    assert rendered.value.outputs[0].name == "training_figure:loss"
    figure = _built(record)
    _, ys = _xy(figure.axes[0].lines[0])
    assert ys.tolist() == [0.0, 1.0]
    figure.clf()


def test_duplicate_x_positions_and_one_point_series_stay_visible() -> None:
    record = _record(
        (
            _series("canonical train", (1.0, 1.0, 2.0), (2.0, 3.0, 1.0)),
            _series("validation", (2.0,), (1.5,)),
        ),
    )
    figure = _built(record)
    axes = figure.axes[0]
    xs, _ = _xy(axes.lines[0])
    assert xs.tolist() == [1.0, 1.0, 2.0]
    singleton = axes.lines[1]
    singleton_xs, _ = _xy(singleton)
    assert singleton_xs.tolist() == [2.0]
    assert singleton.get_marker() == "o"
    figure.clf()


def test_points_only_renders_scatter_never_a_line() -> None:
    record = _record(
        (
            _series("validation", (1.0, 2.0), (2.0, 2.5)),
            _series(
                "selected-checkpoint test",
                (2.0,),
                (3.0,),
                points_only=True,
            ),
        ),
    )
    figure = _built(record)
    axes = figure.axes[0]
    scatters = [artist for artist in axes.collections if isinstance(artist, PathCollection)]
    assert len(scatters) == 1
    offsets = np.asarray(scatters[0].get_offsets(), dtype=np.float64)
    assert offsets.shape == (1, 2)
    assert offsets[0].tolist() == [2.0, 3.0]
    assert all(not str(line.get_label()).startswith("selected-checkpoint") for line in axes.lines)
    figure.clf()


def test_markers_use_recorded_positions_and_short_hash() -> None:
    record = _record(
        (_series("validation", (1.0, 2.0, 3.0), (2.0, 2.5, 3.0)),),
        markers=(
            TrainingMarker("best checkpoint", 2.0, "1" * 64),
            TrainingMarker("stopping state", 3.0),
        ),
    )
    figure = _built(record)
    assert sorted(_vline_positions(figure)) == [2.0, 3.0]
    labels = " ".join(_legend_texts(figure))
    assert "best checkpoint" in labels
    assert "11111111" in labels
    assert "1" * 64 not in labels
    assert "IMAGE exposures=3" in labels
    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert "captured events" in legend.get_title().get_text()
    assert "synthetic captured population" in _figure_text(figure)
    figure.clf()


def test_non_image_units_label_as_events_not_images() -> None:
    record = _record(
        (_series("parameter group 0", (1.0, 2.0), (0.01, 0.01), unit="BATCH", n=3),),
    )
    figure = _built(record)
    labels = " ".join(_legend_texts(figure))
    assert "BATCH events=3" in labels
    assert "IMAGE" not in labels
    figure.clf()


def test_all_null_linear_renders_unavailable_placeholder() -> None:
    record = _record(
        (_series("canonical train", (1.0, 2.0), (None, None)),),
    )
    rendered = render_training_figures((record,), _CFG)
    assert isinstance(rendered, Ok)
    output = rendered.value.outputs[0]
    assert output.status == "UNAVAILABLE"
    assert output.reason == "No captured points are available"
    figure = _built(record)
    assert "No captured points are available" in _figure_text(figure)
    figure.clf()


@pytest.mark.parametrize(
    "record",
    [
        _record(
            (
                TrainingSeries(
                    "canonical train",
                    (1.0, 2.0),
                    (1.0,),
                    2,
                    "synthetic captured population",
                ),
            ),
        ),
        _record(
            (_series("canonical train", (1.0, 2.0), (1.0, 2.0)),),
            identifier="../escape",
        ),
        _record(
            (_series("canonical train", (1.0, float("nan")), (1.0, 2.0)),),
        ),
        _record(
            (_series("canonical train", (1.0, 2.0), (1.0, float("inf"))),),
        ),
    ],
    ids=["length-mismatch", "unsafe-identifier", "nonfinite-x", "nonfinite-y"],
)
def test_malformed_recipes_fail_closed(record: TrainingFigure) -> None:
    assert isinstance(training_figure(record, _CFG), Err)
    assert matplotlib.pyplot.get_fignums() == []


def test_log_axis_marker_at_zero_is_omitted_with_annotation() -> None:
    record = _record(
        (_series("canonical train", (1.0, 2.0), (2.0, 1.0)),),
        x_scale="log",
        markers=(TrainingMarker("current running state", 0.0),),
    )
    figure = _built(record)
    assert _vline_positions(figure) == []
    text = _figure_text(figure)
    assert "current running state" in text
    assert "logarithmic" in text
    figure.clf()


def test_all_masked_log_domain_renders_unavailable_placeholder() -> None:
    record = _record(
        (_series("canonical train", (1.0, 2.0), (0.0, 0.0)),),
        y_scale="log",
    )
    rendered = render_training_figures((record,), _CFG)
    assert isinstance(rendered, Ok)
    output = rendered.value.outputs[0]
    assert output.status == "UNAVAILABLE"
    assert output.reason == "No captured points lie in the requested logarithmic domain"
    figure = _built(record)
    assert "No captured points lie in the requested logarithmic domain" in _figure_text(figure)
    figure.clf()


def test_unavailable_recipe_indexes_placeholder_with_reason() -> None:
    record = _record(
        (
            _series("before update", (), (), unit="BATCH", n=0),
            _series("after update", (), (), unit="BATCH", n=0),
        ),
        identifier="amp_scale",
        reason="Requested values were not captured or are inactive",
    )
    rendered = render_training_figures((record,), _CFG)
    assert isinstance(rendered, Ok)
    output = rendered.value.outputs[0]
    assert output.name == "training_figure:amp_scale"
    assert output.status == "UNAVAILABLE"
    assert output.reason == "Requested values were not captured or are inactive"
    figure = _built(record)
    assert "Requested values were not captured or are inactive" in _figure_text(figure)
    figure.clf()


def test_incomplete_run_status_and_warnings_stay_visible() -> None:
    record = _record(
        (_series("canonical train", (1.0,), (2.0,)),),
        run_status="INTERRUPTED",
        warnings=("Incomplete prefix",),
    )
    figure = _built(record)
    text = _figure_text(figure)
    assert "INTERRUPTED" in text
    assert "Incomplete prefix" in text
    figure.clf()


def test_render_exports_under_training_namespace_and_closes_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.ml_models.analysis.training as training_module

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("renderer must not read run history")

    monkeypatch.setattr(training_module, "read_training_history", _boom)
    records = (
        _record((_series("canonical train", (1.0, 2.0), (2.0, 1.0)),)),
        _record(
            (_series("before update", (), (), unit="BATCH", n=0),),
            identifier="amp_scale",
            reason="Requested values were not captured or are inactive",
        ),
    )
    rendered = render_training_figures(records, _CFG)
    assert isinstance(rendered, Ok)
    paths = {file.path for file in rendered.value.files}
    assert paths == {
        "figures/training/loss.png",
        "figures/training/loss.svg",
        "figures/training/amp_scale.png",
        "figures/training/amp_scale.svg",
    }
    assert {output.name for output in rendered.value.outputs} == {
        "training_figure:loss",
        "training_figure:amp_scale",
    }
    assert matplotlib.pyplot.get_fignums() == []


def test_render_failure_returns_err_and_closes() -> None:
    record = _record(
        (_series("canonical train", (1.0, 2.0), (2.0, 1.0)),),
        x_scale="not-a-scale",  # type: ignore[arg-type]
    )
    result = training_figure(record, _CFG)
    assert isinstance(result, Err)
    assert matplotlib.pyplot.get_fignums() == []
