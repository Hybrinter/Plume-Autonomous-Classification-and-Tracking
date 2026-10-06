"""Model-figure rendering over frozen model-chart recipes.

Draws exact captured coordinates with recorded draw conventions: ``LINE``
and step variants keep null gaps as line breaks, ``POINT`` renders
markers only, and ``BAR`` heights copy recorded values at the frozen x
coordinates (null heights stay absent, never clamped to zero). Interval
bars use the recipe's absolute lower/upper endpoints (never ``yerr``,
which would assume the estimate is enclosed). Matrix figures mask null
cells, annotate exact support counts with contrast-adaptive text, and
use the supplied ``matrix_range`` or the data bounds. No coordinate is
smoothed, filtered, or recomputed; recipes marked unavailable render
explicit placeholders.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math
import textwrap
from typing import TYPE_CHECKING

import numpy as np
from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.artifacts import BundleFile
from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.contracts import AvailabilityRecord, check_bundle_path
from tools.ml_models.analysis.model_figures import ModelFigure, ModelSeries
from tools.ml_models.analysis.plots.common import export_figure
from tools.ml_models.analysis.plots.dataset import RenderedDatasetFigures

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_SPLIT_COLORS = {
    "train": "#0072B2",
    "val": "#E69F00",
    "selected-checkpoint test": "#009E73",
}
_CLASS_COLORS = {
    "negative truth": "#0072B2",
    "positive truth": "#E69F00",
    "correct": "#009E73",
    "incorrect": "#D55E00",
}
_BASELINE_COLOR = "#777777"
_CYCLE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
_BASELINE_NAMES = frozenset(
    {
        "perfect calibration",
        "chance diagonal",
        "random-selection diagonal",
        "random-selection lift",
        "recorded cohort prevalence",
    }
)
_DRAW_STYLES = frozenset({"LINE", "PRE", "POST", "BAR", "POINT"})


def _wrap(text: str, cfg: PlotConfig) -> str:
    """Wrap caption text to a width bounded by the configured figure size."""
    return textwrap.fill(text, width=max(24, int(float(cfg.width_inches) * 11)))


def _color(series: ModelSeries, index: int) -> str:
    """Pick a deterministic series color; baselines stay neutral dark gray."""
    name = series.name
    if name in _CLASS_COLORS:
        return _CLASS_COLORS[name]
    if name in _BASELINE_NAMES:
        return _BASELINE_COLOR
    return _SPLIT_COLORS.get(name, _CYCLE[index % len(_CYCLE)])


def _is_finite(value: float | None) -> bool:
    """Return True only for present finite values."""
    return value is not None and math.isfinite(value)


def _is_count(value: object) -> bool:
    """Return True only for real nonnegative integers, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate(record: ModelFigure) -> Result[None, str]:
    """Reject malformed shapes/paths before any drawing; never truncate."""
    if record.kind not in ("SERIES", "MATRIX"):
        return Err(f"unknown model figure kind {record.kind!r}")
    if record.kind == "MATRIX":
        if not record.x_categories or not record.y_categories:
            return Err("matrix recipes need recorded x and y categories")
        if len(record.matrix) != len(record.y_categories):
            return Err("matrix row count differs from recorded truth categories")
        if len(record.matrix_support) != len(record.y_categories):
            return Err("matrix support row count differs from recorded truth categories")
        for row, support in zip(record.matrix, record.matrix_support, strict=True):
            if len(row) != len(record.x_categories) or len(support) != len(record.x_categories):
                return Err("matrix column count differs from recorded prediction categories")
            if any(not _is_finite(value) for value in row if value is not None):
                return Err("matrix cells must be finite or explicitly missing")
            if any(not _is_count(value) for value in support):
                return Err("matrix support must be nonnegative counts")
        if record.matrix_range is not None:
            low, high = record.matrix_range
            if not (math.isfinite(low) and math.isfinite(high)) or low > high:
                return Err("recorded matrix range is not an ordered finite bound")
    for series in record.series:
        if series.style not in _DRAW_STYLES:
            return Err(f"series {series.name} has unknown draw style {series.style!r}")
        if len(series.x) != len(series.y):
            return Err(f"series {series.name} coordinates have mismatched lengths")
        if any(not math.isfinite(value) for value in series.x):
            return Err(f"series {series.name} has non-finite x coordinates")
        if any(value is not None and not math.isfinite(value) for value in series.y):
            return Err(f"series {series.name} has non-finite captured values")
        if (
            bool(series.lower) != bool(series.upper)
            or (series.lower and len(series.lower) != len(series.x))
            or (series.upper and len(series.upper) != len(series.x))
        ):
            return Err(f"series {series.name} interval ends are misaligned")
        for edge_low, edge_high in zip(series.lower, series.upper, strict=True):
            if (edge_low is None) != (edge_high is None):
                return Err(f"series {series.name} interval endpoints must pair as nulls")
            if (
                edge_low is not None
                and edge_high is not None
                and (
                    not (math.isfinite(edge_low) and math.isfinite(edge_high))
                    or edge_low > edge_high
                )
            ):
                return Err(f"series {series.name} interval endpoints are not ordered finite")
        if series.point_support and len(series.point_support) != len(series.x):
            return Err(f"series {series.name} per-point support is misaligned")
        if any(not _is_count(value) for value in series.point_support):
            return Err(f"series {series.name} per-point support must be nonnegative counts")
    if record.x_categories and any(
        len(series.x) != len(record.x_categories) for series in record.series
    ):
        return Err("series lengths differ from the recorded category count")
    for point in record.points:
        if not (math.isfinite(point.x) and math.isfinite(point.y)):
            return Err("operating points must have finite recorded coordinates")
    for token in (record.identifier, record.identity.split):
        if not token or "/" in token or "\\" in token or token in (".", ".."):
            return Err("model figure identifiers and splits must be single safe tokens")
    try:
        check_bundle_path(f"figures/f/{record.identity.split}/{record.identifier}.png")
    except (ValueError, TypeError) as exc:
        return Err(f"model figure identity is not a safe bundle path: {exc}")
    return Ok(None)


def _series_label(series: ModelSeries) -> str:
    """Name the series with its support unit and count."""
    return f"{series.name} ({series.support.unit} n={series.support.n})"


def _draw_series(axes: Axes, series: ModelSeries, index: int) -> int:
    """Draw one series verbatim; returns the number of masked missing points."""
    xs = np.asarray(series.x, dtype=float)
    ys = np.asarray([value if value is not None else np.nan for value in series.y], dtype=float)
    color = _color(series, index)
    baseline = series.name in _BASELINE_NAMES
    zorder = 1 if baseline else 2
    linestyle = "--" if baseline else "-"
    missing = int(np.isnan(ys).sum())
    if series.style == "BAR":
        axes.bar(
            xs,
            ys,
            color=color,
            width=0.75 * _bar_width(xs),
            label=_series_label(series),
            zorder=zorder,
        )
        for x, value in zip(xs, ys, strict=True):
            if np.isnan(value):
                axes.text(
                    x,
                    0.0,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize="small",
                    color="#666666",
                )
    elif series.style == "POINT":
        axes.plot(
            xs,
            ys,
            linestyle="None",
            marker="o",
            ms=5,
            color=color,
            label=_series_label(series),
            zorder=zorder,
        )
    else:
        drawstyle = {"LINE": "default", "PRE": "steps-pre", "POST": "steps-post"}[series.style]
        axes.plot(
            xs,
            ys,
            drawstyle=drawstyle,
            marker="o",
            ms=3,
            lw=1.3,
            linestyle=linestyle,
            color=color,
            label=_series_label(series),
            zorder=zorder,
        )
    if series.lower and series.upper:
        keep = [
            i for i in range(len(xs)) if series.lower[i] is not None and series.upper[i] is not None
        ]
        if keep:
            low = np.asarray([series.lower[i] for i in keep], dtype=float)
            high = np.asarray([series.upper[i] for i in keep], dtype=float)
            axes.vlines(xs[keep], low, high, color=color, lw=1.0, alpha=0.7)
            axes.plot(xs[keep], low, linestyle="None", marker="_", color=color)
            axes.plot(xs[keep], high, linestyle="None", marker="_", color=color)
    if series.point_support:
        for i, count in enumerate(series.point_support):
            anchor = ys[i] if not np.isnan(ys[i]) else 0.0
            axes.annotate(
                f"n={count}",
                (xs[i], anchor),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize="x-small",
                color="#444444",
            )
    return missing


def _bar_width(xs: np.ndarray) -> float:
    """Scale the bar half-spacing from the frozen coordinate spacing."""
    if len(xs) < 2:
        return 1.0
    gaps = np.diff(np.unique(xs))
    return float(gaps.min()) if len(gaps) else 1.0


def _draw_matrix(axes: Axes, record: ModelFigure, cfg: PlotConfig) -> None:
    """Draw a recorded matrix with masked unavailable cells and exact counts."""
    import matplotlib

    raw = np.asarray(
        [[value if value is not None else np.nan for value in row] for row in record.matrix],
        dtype=float,
    )
    masked = np.ma.masked_invalid(raw)
    cmap = matplotlib.colormaps["viridis"].with_extremes(bad="#BBBBBB")
    if record.matrix_range is not None:
        low, high = record.matrix_range
    elif masked.count():
        low, high = float(masked.min()), float(masked.max())
        if low == high:
            low, high = low - 0.5, high + 0.5
    else:
        low, high = 0.0, 1.0
    image = axes.imshow(masked, cmap=cmap, vmin=low, vmax=high, aspect="equal")
    for row_index, row in enumerate(record.matrix):
        for column_index, value in enumerate(row):
            count = record.matrix_support[row_index][column_index]
            if value is None:
                axes.text(
                    column_index,
                    row_index,
                    "n/a",
                    ha="center",
                    va="center",
                    fontsize="small",
                    color="#333333",
                )
            else:
                fraction = (value - low) / (high - low) if high > low else 0.5
                rgba = cmap(fraction)
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                axes.text(
                    column_index,
                    row_index,
                    f"{value:.4g}\nn={count}",
                    ha="center",
                    va="center",
                    fontsize="small",
                    color="#FFFFFF" if luminance < 0.45 else "#111111",
                )
    ticks_x = np.arange(len(record.x_categories))
    ticks_y = np.arange(len(record.y_categories))
    axes.set_xticks(ticks_x)
    axes.set_yticks(ticks_y)
    long_labels = any(len(label) > 8 for label in record.x_categories)
    axes.set_xticklabels(
        [textwrap.fill(label, 14) for label in record.x_categories],
        rotation=30 if long_labels else 0,
        ha="right" if long_labels else "center",
    )
    axes.set_yticklabels([textwrap.fill(label, 14) for label in record.y_categories])
    axes.set_xlabel(_wrap(record.x_label, cfg))
    axes.set_ylabel(_wrap(record.y_label, cfg))
    axes.figure.colorbar(image, ax=axes, fraction=0.05)


def _footer_text(record: ModelFigure, missing: int, cfg: PlotConfig) -> str:
    """Compose the wrapped population/notes/missing-count caption."""
    parts = [record.population]
    parts.extend(record.notes)
    if missing:
        parts.append(f"Missing captured points: {missing}")
    return _wrap(" | ".join(part for part in parts if part), cfg)


def _close(figure: Figure | None) -> None:
    """Clear and close a partially built figure after a render failure."""
    if figure is not None:
        figure.clf()
        import matplotlib.pyplot as plt

        plt.close(figure)


def _placeholder(record: ModelFigure, cfg: PlotConfig) -> Result[Figure, str]:
    """Render an explicit unavailable panel carrying the recorded reason."""
    import matplotlib

    figure: Figure | None = None
    try:
        matplotlib.use("Agg")
        with matplotlib.rc_context({"font.size": float(cfg.font_size)}):
            from matplotlib.figure import Figure

            figure = Figure(
                figsize=(float(cfg.width_inches), float(cfg.height_inches)),
                dpi=int(cfg.dpi),
            )
            figure.set_layout_engine("constrained")
            figure.suptitle(_wrap(record.title, cfg))
            axes = figure.add_subplot()
            axes.set_xticks(())
            axes.set_yticks(())
            axes.text(
                0.5,
                0.5,
                _wrap(record.reason or "Unavailable", cfg),
                ha="center",
                va="center",
                transform=axes.transAxes,
                color="#555555",
            )
            figure.supxlabel(_footer_text(record, 0, cfg), fontsize="x-small", color="#444444")
            return Ok(figure)
    except (OSError, ValueError, RuntimeError) as exc:
        _close(figure)
        return Err(f"cannot render {record.identifier} placeholder: {exc}")


def model_figure(record: ModelFigure, cfg: PlotConfig) -> Result[Figure, str]:
    """Render one frozen model recipe; never smooths, fits, or recomputes."""
    import matplotlib

    valid = _validate(record)
    if isinstance(valid, Err):
        return valid
    if record.reason is not None:
        return _placeholder(record, cfg)
    figure: Figure | None = None
    try:
        matplotlib.use("Agg")
        with matplotlib.rc_context({"font.size": float(cfg.font_size)}):
            from matplotlib.figure import Figure

            figure = Figure(
                figsize=(float(cfg.width_inches), float(cfg.height_inches)),
                dpi=int(cfg.dpi),
            )
            figure.set_layout_engine("constrained")
            figure.suptitle(_wrap(record.title, cfg))
            axes = figure.add_subplot()
            if record.kind == "MATRIX":
                _draw_matrix(axes, record, cfg)
                figure.supxlabel(_footer_text(record, 0, cfg), fontsize="x-small", color="#444444")
            else:
                missing = 0
                for index, series in enumerate(record.series):
                    missing += _draw_series(axes, series, index)
                for point in record.points:
                    axes.plot(
                        [point.x],
                        [point.y],
                        linestyle="None",
                        marker="*",
                        ms=11,
                        color="#D55E00",
                        mec="#7F3B08",
                        label=point.name,
                    )
                if record.x_categories:
                    aligned = next(
                        (
                            series
                            for series in record.series
                            if len(series.x) == len(record.x_categories)
                        ),
                        None,
                    )
                    if aligned is not None:
                        ticks = np.asarray(aligned.x, dtype=float)
                        axes.set_xticks(ticks)
                        axes.set_xticklabels(
                            [textwrap.fill(label, 14) for label in record.x_categories]
                        )
                        margin = max(0.4, float(ticks.max() - ticks.min()) * 0.1)
                        axes.set_xlim(float(ticks.min()) - margin, float(ticks.max()) + margin)
                axes.set_xlabel(_wrap(record.x_label, cfg))
                axes.set_ylabel(_wrap(record.y_label, cfg))
                handles, labels = axes.get_legend_handles_labels()
                if handles:
                    axes.legend(
                        fontsize="small",
                        title=_wrap(record.population, cfg),
                        title_fontsize="x-small",
                    )
                figure.supxlabel(
                    _footer_text(record, missing, cfg), fontsize="x-small", color="#444444"
                )
            return Ok(figure)
    except (OSError, ValueError, RuntimeError) as exc:
        _close(figure)
        return Err(f"cannot render {record.identifier}: {exc}")


def _safe_family(family: str) -> Result[None, str]:
    """Require a single safe relative token for the figure family."""
    if not family or "/" in family or "\\" in family or family in (".", ".."):
        return Err("figure family must be a single safe relative token")
    try:
        check_bundle_path(f"figures/{family}/placeholder.png")
    except (ValueError, TypeError) as exc:
        return Err(f"unsafe figure family: {exc}")
    return Ok(None)


def render_model_figures(
    figures: tuple[ModelFigure, ...],
    cfg: PlotConfig,
    *,
    family: str,
) -> Result[RenderedDatasetFigures, str]:
    """Export every supplied recipe under ``figures/<family>/<split>/``.

    Figure identifiers are the frozen recipe identifiers; availability is
    copied from each recipe's recorded reason. Recipes from different
    splits may share an identifier; the same split may not carry it
    twice. No figure is regenerated from source evidence.
    """
    safe = _safe_family(family)
    if isinstance(safe, Err):
        return safe
    identities = {(record.identity.split, record.identifier) for record in figures}
    if len(identities) != len(figures):
        return Err("model figure (split, identifier) identities must be unique")
    files: list[BundleFile] = []
    outputs: list[AvailabilityRecord] = []
    for record in figures:
        rendered = model_figure(record, cfg)
        if isinstance(rendered, Err):
            return rendered
        exported = export_figure(
            rendered.value,
            f"{family}/{record.identity.split}/{record.identifier}",
            cfg,
        )
        if isinstance(exported, Err):
            return exported
        files.extend(exported.value)
        name = f"model_figure:{family}:{record.identity.split}:{record.identifier}"
        outputs.append(
            AvailabilityRecord(
                name=name,
                status="UNAVAILABLE" if record.reason else "AVAILABLE",
                reason=record.reason,
            )
        )
    return Ok(RenderedDatasetFigures(files=tuple(files), outputs=tuple(outputs)))
