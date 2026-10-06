"""Tests for frozen dataset figure rendering."""

import struct

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from flight.libs.types import Err, Ok
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.contracts import MetricSupport, SupportUnit
from tools.ml_models.analysis.dataset_figures import DatasetFigure, FigureSeries
from tools.ml_models.analysis.plots.common import export_figure
from tools.ml_models.analysis.plots.dataset import (
    dataset_figure,
    render_dataset_figures,
)

_CFG = PlotConfig(formats=("png", "svg"), dpi=100, width_inches=5.0, height_inches=3.0)


def _series(
    name: str,
    x: tuple[str | float | int, ...],
    y: tuple[float | int | None, ...],
    *,
    unit: SupportUnit = "IMAGE",
    n: int | None = None,
) -> FigureSeries:
    return FigureSeries(name, x, y, MetricSupport(unit=unit, n=len(x) if n is None else n))


def _built(record: DatasetFigure) -> Figure:
    built = dataset_figure(record, _CFG)
    assert isinstance(built, Ok)
    return built.value


def test_bar_aligns_union_categories_and_annotates_null() -> None:
    """Null points are annotated, never drawn as fake zero bars."""
    record = DatasetFigure(
        "counts_bar",
        "BAR",
        "Counts",
        "Field",
        "Count",
        "canonical variants",
        series=(
            _series("train", ("a", "b", "c"), (1, None, 3), n=3),
            _series("val", ("b", "c"), (2, 4), n=2),
        ),
    )
    figure = _built(record)
    axes = figure.axes[0]
    bars = [rect for rect in axes.patches if isinstance(rect, Rectangle)]
    assert sorted(rect.get_height() for rect in bars) == [1, 2, 3, 4]
    assert [label.get_text() for label in axes.get_xticklabels()] == ["a", "b", "c"]
    assert sum(text.get_text() == "n/a" for text in axes.texts) == 1
    legend = axes.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "canonical variants"
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["train (IMAGE n=3)", "val (IMAGE n=2)"]
    colors = {rect.get_facecolor() for rect in axes.patches}
    assert len(colors) == 2


def test_ecdf_draws_post_steps_and_keeps_empty_series_in_legend() -> None:
    """Exact step coordinates; zero-support series stay visible as unavailable."""
    record = DatasetFigure(
        "gsd_ecdf",
        "ECDF",
        "GSD coverage",
        "GSD (m)",
        "Empirical cumulative fraction",
        "canonical variants",
        series=(
            _series("train", (10.0, 20.0), (0.5, 1.0), n=2),
            _series("val", (), (), n=0),
        ),
    )
    figure = _built(record)
    axes = figure.axes[0]
    data_lines = [line for line in axes.get_lines() if np.asarray(line.get_xdata()).size]
    assert len(data_lines) == 1
    assert np.asarray(data_lines[0].get_xdata()).tolist() == [10.0, 20.0]
    assert np.asarray(data_lines[0].get_ydata()).tolist() == [0.5, 1.0]
    legend = axes.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels[0] == "train (IMAGE n=2)"
    assert "val" in labels[1] and "n=0" in labels[1] and "unavailable" in labels[1]


def test_ecdf_singleton_series_stays_visible_with_marker() -> None:
    """A one-point ECDF keeps its exact coordinate and a visible marker."""
    record = DatasetFigure(
        "gsd_ecdf",
        "ECDF",
        "GSD coverage",
        "GSD (m)",
        "Empirical cumulative fraction",
        "canonical variants",
        series=(
            _series("train", (10.0, 20.0), (0.5, 1.0), n=2),
            _series("val", (16.5,), (1.0,), n=1),
        ),
    )
    figure = _built(record)
    lines = [line for line in figure.axes[0].get_lines() if np.asarray(line.get_xdata()).size]
    singleton = next(line for line in lines if np.asarray(line.get_xdata()).size == 1)
    assert singleton.get_marker() == "o"
    assert np.asarray(singleton.get_xdata()).tolist() == [16.5]
    assert np.asarray(singleton.get_ydata()).tolist() == [1.0]


def test_histogram_uses_supplied_counts_and_edges_without_rebinning() -> None:
    """Stair vertices carry exactly the frozen counts and edges."""
    record = DatasetFigure(
        "pixel_histogram_red",
        "HISTOGRAM",
        "RED distribution",
        "Processed unit value",
        "Pixel count",
        "all pixels",
        series=(_series("RED", (0.0, 0.5, 1.0), (2, 4), unit="PIXEL", n=6),),
    )
    figure = _built(record)
    patch = figure.axes[0].patches[0]
    vertices = np.asarray(patch.get_path().vertices)
    assert set(np.unique(vertices[:, 0])) == {0.0, 0.5, 1.0}
    assert set(np.unique(vertices[:, 1])) == {0.0, 2.0, 4.0}


def test_matrix_masks_null_pairs_and_fixes_pearson_scale() -> None:
    """Constant-band nulls are masked, labelled n/a, on a fixed -1..1 scale."""
    record = DatasetFigure(
        "pixel_correlations",
        "MATRIX",
        "Band correlations",
        "Band",
        "Band",
        "all canonical-variant pixels",
        matrix=((1.0, None), (None, 0.5)),
        matrix_labels=("RED", "NIR"),
        matrix_support=MetricSupport(unit="PIXEL", n=10),
    )
    figure = _built(record)
    axes = figure.axes[0]
    image = axes.images[0]
    assert image.get_clim() == (-1.0, 1.0)
    array = image.get_array()
    assert array is not None
    masked = np.ma.getmaskarray(array)
    assert masked.tolist() == [[False, True], [True, False]]
    texts = [text.get_text() for text in axes.texts]
    assert texts.count("n/a") == 2
    assert "1.00" in texts and "0.50" in texts
    assert [label.get_text() for label in axes.get_xticklabels()] == ["RED", "NIR"]
    assert [label.get_text() for label in axes.get_yticklabels()] == ["RED", "NIR"]
    colorbar_axes = figure.axes[-1]
    assert colorbar_axes.get_ylabel() == "Pearson correlation"
    assert "PIXEL n=10" in " ".join(axes.get_title().split())


def test_unavailable_recipe_renders_reason_placeholder() -> None:
    """A reason-only recipe is a labelled placeholder with no fabricated data."""
    record = DatasetFigure(
        "conditions_unavailable",
        "BAR",
        "Recorded condition coverage",
        "Condition",
        "Count",
        "recorded categorical conditions",
        reason="No categorical conditions were recorded",
    )
    figure = _built(record)
    axes = figure.axes[0]
    texts = [text.get_text() for text in axes.texts]
    assert any("No categorical conditions were recorded" in text for text in texts)
    assert not axes.patches and not axes.get_lines()


def test_export_figure_emits_exact_formats_and_png_dimensions() -> None:
    """PNG pixels equal configured inches times dpi; paths use the kind prefix."""
    figure = _built(DatasetFigure("f", "BAR", "t", "x", "y", "pop", reason="none available"))
    exported = export_figure(figure, "my_fig", _CFG)
    assert isinstance(exported, Ok)
    files = exported.value
    assert [file.path for file in files] == ["figures/my_fig.png", "figures/my_fig.svg"]
    png = files[0].data
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", png[16:24]) == (500, 300)
    assert files[1].data.lstrip().startswith(b"<?xml")
    visual = export_figure(
        _built(DatasetFigure("v", "BAR", "t", "x", "y", "pop", reason="r")),
        "gal",
        _CFG,
        kind="VISUAL",
    )
    assert isinstance(visual, Ok)
    assert all(file.path.startswith("visuals/gal.") for file in visual.value)


def test_render_dataset_figures_exports_and_indexes_every_recipe() -> None:
    """Outputs name every recipe; unavailable recipes keep bytes and reasons."""
    records = (
        DatasetFigure(
            "ok_bar",
            "BAR",
            "t",
            "x",
            "y",
            "pop",
            series=(_series("train", ("a",), (1,)),),
        ),
        DatasetFigure("missing", "BAR", "t", "x", "y", "pop", reason="no data"),
    )
    rendered = render_dataset_figures(records, _CFG)
    assert isinstance(rendered, Ok)
    assert {file.path for file in rendered.value.files} == {
        "figures/ok_bar.png",
        "figures/ok_bar.svg",
        "figures/missing.png",
        "figures/missing.svg",
    }
    outputs = {record.name: record for record in rendered.value.outputs}
    assert outputs["figure:ok_bar"].status == "AVAILABLE"
    assert outputs["figure:missing"].status == "UNAVAILABLE"
    assert outputs["figure:missing"].reason == "no data"
    assert plt.get_fignums() == []


def test_failed_savefig_still_closes_the_figure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A codec failure returns Err and leaves no open figure handles."""
    figure = _built(DatasetFigure("f", "BAR", "t", "x", "y", "pop", reason="r"))

    def boom(self: Figure, *args: object, **kwargs: object) -> None:
        raise OSError("simulated savefig failure")

    monkeypatch.setattr(Figure, "savefig", boom)
    result = export_figure(figure, "f", _CFG)
    assert isinstance(result, Err)
    assert figure.axes == []
    assert plt.get_fignums() == []


def test_rendering_never_reads_datasets_or_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendering consumes frozen recipes only; source/model loaders are uncalled."""
    import tools.ml_models.analysis.dataset as dataset_module
    import tools.ml_models.dataset.store as store_module

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("source/model I/O during rendering")

    monkeypatch.setattr(dataset_module, "measure_dataset", boom)
    monkeypatch.setattr(store_module, "read_rows", boom)
    record = DatasetFigure(
        "f",
        "BAR",
        "t",
        "x",
        "y",
        "pop",
        series=(_series("train", ("a",), (1,)),),
    )
    rendered = render_dataset_figures((record,), _CFG)
    assert isinstance(rendered, Ok)


def test_export_resizes_caller_figure_to_configured_dimensions() -> None:
    """A differently-sized supplied figure still emits cfg inches times dpi."""
    import matplotlib.figure

    figure = matplotlib.figure.Figure(figsize=(1.0, 1.0), dpi=50)
    figure.add_subplot().plot([0, 1], [0, 1])
    exported = export_figure(
        figure,
        "resized",
        PlotConfig(formats=("png",), dpi=100, width_inches=5.0, height_inches=3.0),
    )
    assert isinstance(exported, Ok)
    assert struct.unpack(">II", exported.value[0].data[16:24]) == (500, 300)


def test_export_leaves_rcparams_and_rejects_unknown_kind() -> None:
    """rc_context keeps global rcParams stable; non-figure kinds are refused."""
    record = DatasetFigure("f", "BAR", "t", "x", "y", "pop", reason="r")
    before_salt = matplotlib.rcParams["svg.hashsalt"]
    before_font = matplotlib.rcParams["font.size"]
    exported = export_figure(_built(record), "ok", _CFG)
    assert isinstance(exported, Ok)
    assert matplotlib.rcParams["svg.hashsalt"] == before_salt
    assert matplotlib.rcParams["font.size"] == before_font
    rejected = export_figure(_built(record), "bad", _CFG, kind="TABLE")
    assert isinstance(rejected, Err)


def test_svg_bytes_are_deterministic_for_identical_content() -> None:
    """The fixed hash salt and omitted date give repeatable SVG bytes."""
    record = DatasetFigure(
        "repeat",
        "BAR",
        "t",
        "x",
        "y",
        "pop",
        series=(_series("train", ("a",), (1,)),),
    )
    first = export_figure(_built(record), "repeat", PlotConfig(formats=("svg",), dpi=100))
    second = export_figure(_built(record), "repeat", PlotConfig(formats=("svg",), dpi=100))
    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert first.value[0].data == second.value[0].data
