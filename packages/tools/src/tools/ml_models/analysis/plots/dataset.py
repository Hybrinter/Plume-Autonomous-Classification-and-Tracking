"""Dataset-analysis figure rendering over frozen chart recipes.

Each ``DatasetFigure`` recipe from ``analysis.dataset_figures`` is drawn
verbatim: coordinates, support, and availability are already frozen, so
rendering never recomputes bins, CDFs, or rates and never reads source
tensors, datasets, or models. Output arrives as typed ``BundleFile``
bytes through ``export_figure``; no user directory is reserved here.
Unavailable recipes render as labelled placeholders and are still
indexed.

Contains:
  - RenderedDatasetFigures: exported bundle files plus per-figure
    availability records.
  - dataset_figure: build one in-memory matplotlib Figure per recipe.
  - render_dataset_figures: export every recipe into bundle bytes.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.artifacts import BundleFile
from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.contracts import AvailabilityRecord
from tools.ml_models.analysis.dataset_figures import DatasetFigure, FigureSeries
from tools.ml_models.analysis.plots.common import export_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_SPLIT_COLORS: dict[str, str] = {
    "train": "#0072B2",
    "val": "#E69F00",
    "test": "#009E73",
}


@dataclass(frozen=True, slots=True)
class RenderedDatasetFigures:
    """Bundle bytes and availability for one dataset figure inventory.

    Attributes:
        files: Exported figure bytes under ``figures/{identifier}.{format}``.
        outputs: One ``figure:{identifier}`` availability record per recipe.
    """

    files: tuple[BundleFile, ...]
    outputs: tuple[AvailabilityRecord, ...]


def _series_color(name: str, index: int) -> str:
    """Use fixed split colours; other series follow the default cycle."""
    return _SPLIT_COLORS.get(name, f"C{index}")


def _series_label(series: FigureSeries) -> str:
    """Name the series with its support unit and count; empty stays explicit."""
    label = f"{series.name} ({series.support.unit} n={series.support.n})"
    if series.support.n == 0 or not series.x:
        return label + ", unavailable"
    return label


def _wrap(text: str, cfg: PlotConfig, scale: float = 11.0) -> str:
    """Wrap prose to a width bounded by the configured figure width."""
    return textwrap.fill(text, width=max(24, int(float(cfg.width_inches) * scale)))


def _population(axes: Axes, record: DatasetFigure, cfg: PlotConfig) -> None:
    """Draw the labelled population as a wrapped legend title."""
    if record.series:
        axes.legend(title=textwrap.fill(record.population, 30))


def _headings(axes: Axes, record: DatasetFigure, cfg: PlotConfig) -> None:
    """Apply the wrapped recorded title and axis labels."""
    axes.set_title(_wrap(record.title, cfg))
    axes.set_xlabel(record.x_label)
    axes.set_ylabel(record.y_label)


def _placeholder(axes: Axes, record: DatasetFigure, cfg: PlotConfig) -> None:
    """Render an unavailable recipe as a labelled empty panel."""
    _headings(axes, record, cfg)
    axes.set_xticks(())
    axes.set_yticks(())
    axes.text(
        0.5,
        0.5,
        _wrap(f"Unavailable: {record.reason}\n{record.population}", cfg),
        ha="center",
        va="center",
        transform=axes.transAxes,
        fontsize=float(cfg.font_size) * 0.9,
        color="#666666",
    )


def _bar(axes: Axes, record: DatasetFigure, cfg: PlotConfig) -> None:
    """Draw grouped categorical bars; null points are annotated, never zeroed."""
    categories: list[str | float | int] = []
    for series in record.series:
        for value in series.x:
            if value not in categories:
                categories.append(value)
    count = max(len(record.series), 1)
    width = 0.8 / count
    for index, series in enumerate(record.series):
        offset = (index - (count - 1) / 2) * width
        lookup = dict(zip(series.x, series.y, strict=True))
        xs: list[float] = []
        heights: list[float] = []
        for value in series.x:
            height = lookup[value]
            if height is None:
                continue
            xs.append(categories.index(value) + offset)
            heights.append(float(height))
        axes.bar(
            xs,
            heights,
            width,
            color=_series_color(series.name, index),
            label=_series_label(series),
        )
        for value in series.x:
            if lookup[value] is None:
                axes.annotate(
                    "n/a",
                    (categories.index(value) + offset, 0.0),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=float(cfg.font_size) * 0.7,
                    color="#666666",
                )
    labels = [str(value) for value in categories]
    if max((len(label) for label in labels), default=0) > 14:
        axes.set_xticks(
            range(len(categories)),
            labels=[textwrap.fill(label, 14) for label in labels],
            rotation=30,
            ha="right",
        )
    else:
        axes.set_xticks(range(len(categories)), labels=labels)
    axes.set_xlim(-0.6, len(categories) - 0.4)
    _population(axes, record, cfg)


def _ecdf_plot(axes: Axes, record: DatasetFigure, cfg: PlotConfig) -> None:
    """Draw exact right-continuous ECDF steps; empty series stay in the legend."""
    for index, series in enumerate(record.series):
        label = _series_label(series)
        color = _series_color(series.name, index)
        if series.x and all(value is not None for value in series.y):
            xs = [float(value) for value in series.x]
            ys = [float(value) for value in series.y if value is not None]
            axes.step(xs, ys, where="post", label=label, color=color, marker="o")
        else:
            axes.step((), (), where="post", label=label, color=color)
    axes.set_ylim(0.0, 1.0)
    _population(axes, record, cfg)


def _histogram(axes: Axes, record: DatasetFigure, cfg: PlotConfig) -> None:
    """Draw captured histogram counts over supplied edges without rebinning."""
    for index, series in enumerate(record.series):
        counts = tuple(float(value) for value in series.y if value is not None)
        if len(series.x) != len(counts) + 1:
            raise ValueError(
                f"histogram {record.identifier} has {len(series.x)} edges for {len(counts)} counts"
            )
        axes.stairs(
            counts, series.x, label=_series_label(series), color=_series_color(series.name, index)
        )
    _population(axes, record, cfg)


def _matrix(figure: Figure, axes: Axes, record: DatasetFigure, cfg: PlotConfig) -> None:
    """Render the captured correlation matrix verbatim; null cells are masked."""
    import matplotlib

    values = np.array(
        [[np.nan if value is None else float(value) for value in row] for row in record.matrix],
        dtype=np.float64,
    )
    masked = np.ma.masked_invalid(values)
    cmap = matplotlib.colormaps["coolwarm"].with_extremes(bad="#d0d0d0")
    image = axes.imshow(masked, cmap=cmap, vmin=-1.0, vmax=1.0)
    labels = tuple(str(label) for label in record.matrix_labels)
    axes.set_xticks(range(len(labels)), labels=labels, rotation=90)
    axes.set_yticks(range(len(labels)), labels=labels)
    for row_index, row in enumerate(record.matrix):
        for col_index, value in enumerate(row):
            axes.text(
                col_index,
                row_index,
                "n/a" if value is None else f"{float(value):.2f}",
                ha="center",
                va="center",
                fontsize=float(cfg.font_size) * 0.8,
                color="#333333",
            )
    colorbar = figure.colorbar(image, ax=axes)
    colorbar.set_label("Pearson correlation")
    if record.matrix_support is not None:
        support = record.matrix_support
        title = f"{record.title}\n{record.population} ({support.unit} n={support.n})"
    else:
        title = f"{record.title}\n{record.population}"
    axes.set_title(_wrap(title, cfg))


def dataset_figure(record: DatasetFigure, cfg: PlotConfig) -> Result[Figure, str]:
    """Build one in-memory figure for a frozen recipe; the caller exports and closes."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    figure: Figure | None = None
    try:
        with matplotlib.rc_context({"font.size": float(cfg.font_size)}):
            figure = Figure(
                figsize=(float(cfg.width_inches), float(cfg.height_inches)),
                dpi=int(cfg.dpi),
            )
            figure.set_layout_engine("constrained")
            axes = figure.add_subplot()
            if record.reason is not None:
                _placeholder(axes, record, cfg)
            elif record.kind == "BAR":
                _bar(axes, record, cfg)
                _headings(axes, record, cfg)
            elif record.kind == "ECDF":
                _ecdf_plot(axes, record, cfg)
                _headings(axes, record, cfg)
            elif record.kind == "HISTOGRAM":
                _histogram(axes, record, cfg)
                _headings(axes, record, cfg)
            elif record.kind == "MATRIX":
                _matrix(figure, axes, record, cfg)
                axes.set_xlabel(record.x_label)
                axes.set_ylabel(record.y_label)
            else:
                raise ValueError(f"unsupported figure kind {record.kind!r}")
            return Ok(figure)
    except (OSError, ValueError, RuntimeError, ZeroDivisionError) as exc:
        if figure is not None:
            figure.clf()
            import matplotlib.pyplot as plt

            plt.close(figure)
        return Err(f"cannot render figure {record.identifier}: {exc}")


def render_dataset_figures(
    figures: tuple[DatasetFigure, ...], cfg: PlotConfig
) -> Result[RenderedDatasetFigures, str]:
    """Export every frozen dataset recipe into per-format bundle bytes.

    A recipe carrying ``reason`` renders as a labelled placeholder, is
    exported like any other figure, and is indexed ``UNAVAILABLE``; all
    other figures are ``AVAILABLE``.
    """
    files: list[BundleFile] = []
    outputs: list[AvailabilityRecord] = []
    for record in figures:
        rendered = dataset_figure(record, cfg)
        if isinstance(rendered, Err):
            return rendered
        exported = export_figure(rendered.value, record.identifier, cfg)
        if isinstance(exported, Err):
            return exported
        files.extend(exported.value)
        outputs.append(
            AvailabilityRecord(
                name="figure:" + record.identifier,
                status="UNAVAILABLE" if record.reason is not None else "AVAILABLE",
                reason=record.reason,
            )
        )
    return Ok(RenderedDatasetFigures(files=tuple(files), outputs=tuple(outputs)))
