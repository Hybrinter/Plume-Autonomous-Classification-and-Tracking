"""Training-history figure rendering over frozen chart recipes.

Each ``TrainingFigure`` recipe from ``analysis.training_figures`` is
drawn verbatim: coordinates, markers, run status, and availability are
already frozen, so rendering never re-averages losses, re-selects
checkpoints, interpolates, or reads histories, models, or datasets.
Logarithmic axes mask absent and nonpositive values as NaN gaps at the
same indices; raw zeros stay visible on linear axes. Output arrives as
typed ``BundleFile`` bytes under ``figures/training/`` through
``export_figure``; no user directory is reserved here. Unavailable
recipes render as labelled placeholders and are still indexed.

Contains:
  - training_figure: build one in-memory matplotlib Figure per recipe.
  - render_training_figures: export every recipe into bundle bytes.

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
from tools.ml_models.analysis.contracts import (
    AvailabilityRecord,
    check_bundle_path,
)
from tools.ml_models.analysis.plots.common import export_figure
from tools.ml_models.analysis.plots.dataset import RenderedDatasetFigures
from tools.ml_models.analysis.training_figures import (
    TrainingFigure,
    TrainingSeries,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_SERIES_COLORS: dict[str, str] = {
    "canonical train": "#0072B2",
    "validation": "#E69F00",
    "selected-checkpoint test": "#009E73",
}
_OPTIMIZATION_COLOR = "#555555"
_LOG_DOMAIN_REASON = "No captured points lie in the requested logarithmic domain"
_NO_CAPTURED_REASON = "No captured points are available"
_VALID_SCALES = ("linear", "log")


def _wrap(text: str, cfg: PlotConfig, scale: float = 11.0) -> str:
    """Wrap prose to a width bounded by the configured figure width."""
    return textwrap.fill(text, width=max(24, int(float(cfg.width_inches) * scale)))


def _validate(record: TrainingFigure) -> str | None:
    """Reject unsafe identifiers, unknown scales, and malformed series."""
    try:
        check_bundle_path(f"figures/training/{record.identifier}.png")
    except ValueError as exc:
        return f"unsafe figure identifier {record.identifier!r}: {exc}"
    if record.x_scale not in _VALID_SCALES or record.y_scale not in _VALID_SCALES:
        return (
            f"figure {record.identifier} has unknown scales {record.x_scale!r}/{record.y_scale!r}"
        )
    for series in record.series:
        if len(series.x) != len(series.y):
            return (
                f"figure {record.identifier} series {series.name!r} has "
                f"{len(series.x)} x values for {len(series.y)} y values"
            )
        if any(not math.isfinite(value) for value in series.x):
            return f"figure {record.identifier} series {series.name!r} has nonfinite x"
        if any(value is not None and not math.isfinite(value) for value in series.y):
            return f"figure {record.identifier} series {series.name!r} has nonfinite y"
    if any(not math.isfinite(marker.x) for marker in record.markers):
        return f"figure {record.identifier} has a nonfinite marker position"
    return None


def _masked(
    series: TrainingSeries, record: TrainingFigure
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return float coordinates with nulls and log-domain gaps as NaN.

    Index alignment is preserved so masked positions break the drawn
    line. The second pair of counts distinguishes source-null points
    from points omitted because a logarithmic axis cannot show them.
    """
    xs = np.asarray(series.x, dtype=np.float64)
    ys = np.asarray(
        [np.nan if value is None else float(value) for value in series.y],
        dtype=np.float64,
    )
    unavailable = int(np.isnan(ys).sum())
    finite_y = ~np.isnan(ys)
    outside = np.zeros(xs.shape, dtype=bool)
    if record.x_scale == "log":
        outside |= finite_y & (xs <= 0.0)
    if record.y_scale == "log":
        outside |= finite_y & (ys <= 0.0)
    xs[outside] = np.nan
    ys[outside] = np.nan
    return xs, ys, unavailable, int(outside.sum())


def _point_state(record: TrainingFigure) -> tuple[bool, int]:
    """Return whether any finite pair survives, plus log-omitted point count."""
    plottable = False
    omitted = 0
    for series in record.series:
        xs, ys, _, outside = _masked(series, record)
        plottable = plottable or bool((np.isfinite(xs) & np.isfinite(ys)).any())
        omitted += outside
    return plottable, omitted


def _effective_reason(record: TrainingFigure) -> str | None:
    """Return the recorded reason or the explicit placeholder reason.

    The logarithmic-domain reason applies only when a log axis exists
    and finite captured points were excluded by that domain; otherwise
    an empty figure reports that no captured points are available.
    """
    if record.reason is not None:
        return record.reason
    plottable, omitted = _point_state(record)
    if plottable:
        return None
    if omitted > 0 and (record.x_scale == "log" or record.y_scale == "log"):
        return _LOG_DOMAIN_REASON
    return _NO_CAPTURED_REASON


def _series_label(series: TrainingSeries, unavailable: int, outside: int) -> str:
    """Name the series with event-scoped exposure counts; omit counts stay explicit."""
    if series.exposure_unit == "IMAGE":
        label = f"{series.name} (IMAGE exposures={series.n_exposures})"
    else:
        label = f"{series.name} ({series.exposure_unit} events={series.n_exposures})"
    extras: list[str] = []
    if not series.x:
        extras.append("0 captured points")
    if unavailable:
        extras.append(f"{unavailable} unavailable")
    if outside:
        extras.append(f"{outside} outside logarithmic domain")
    if extras:
        label += "; " + ", ".join(extras)
    return label


def _headings(axes: Axes, record: TrainingFigure, cfg: PlotConfig) -> None:
    """Apply the recorded title, labels, and exact axis scales."""
    axes.set_title(_wrap(record.title, cfg))
    axes.set_xlabel(record.x_label)
    axes.set_ylabel(record.y_label)
    axes.set_xscale(record.x_scale)
    axes.set_yscale(record.y_scale)


def _draw_series(axes: Axes, record: TrainingFigure) -> None:
    """Draw raw coordinates verbatim; masked gaps break lines, nothing bridges."""
    for series in record.series:
        xs, ys, unavailable, outside = _masked(series, record)
        color = _SERIES_COLORS.get(series.name, _OPTIMIZATION_COLOR)
        label = _series_label(series, unavailable, outside)
        if series.points_only:
            keep = np.isfinite(xs) & np.isfinite(ys)
            axes.scatter(xs[keep], ys[keep], marker="o", color=color, label=label)
        else:
            axes.plot(xs, ys, marker="o", color=color, label=label, linewidth=1.2)


def _draw_markers(axes: Axes, record: TrainingFigure) -> list[str]:
    """Draw recorded checkpoint/state markers; return log-omitted names."""
    omitted: list[str] = []
    for marker in record.markers:
        if record.x_scale == "log" and marker.x <= 0.0:
            omitted.append(marker.name)
            continue
        label = marker.name
        if marker.checkpoint_hash is not None:
            label += f" ({marker.checkpoint_hash[:8]})"
        axes.axvline(marker.x, color="#777777", linestyle="--", linewidth=1.0, label=label)
    return omitted


def _footer(figure: Figure, record: TrainingFigure, cfg: PlotConfig, omitted: list[str]) -> None:
    """Publish run status, populations, warnings, and omitted markers as a footer."""
    populations: list[str] = []
    for series in record.series:
        if series.population not in populations:
            populations.append(series.population)
    parts = [f"run status: {record.run_status}"]
    if populations:
        parts.append("populations: " + "; ".join(populations))
    if omitted:
        parts.append("markers outside logarithmic domain: " + ", ".join(omitted))
    parts.extend(record.warnings)
    figure.supxlabel(
        _wrap("; ".join(parts), cfg, scale=14.0),
        fontsize=float(cfg.font_size) * 0.75,
        color="#555555",
    )


def training_figure(record: TrainingFigure, cfg: PlotConfig) -> Result[Figure, str]:
    """Build one in-memory figure for a frozen recipe; the caller exports and closes."""
    import matplotlib

    problem = _validate(record)
    if problem is not None:
        return Err(f"cannot render figure {record.identifier}: {problem}")
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
            _headings(axes, record, cfg)
            reason = _effective_reason(record)
            omitted: list[str] = []
            if reason is not None:
                axes.set_xticks(())
                axes.set_yticks(())
                axes.text(
                    0.5,
                    0.5,
                    _wrap(f"Unavailable: {reason}", cfg),
                    ha="center",
                    va="center",
                    transform=axes.transAxes,
                    fontsize=float(cfg.font_size) * 0.9,
                    color="#666666",
                )
            else:
                _draw_series(axes, record)
                omitted = _draw_markers(axes, record)
                axes.legend(
                    fontsize=float(cfg.font_size) * 0.75,
                    title=textwrap.fill("counts across captured events, not unique images", 30),
                    title_fontsize=float(cfg.font_size) * 0.7,
                )
            _footer(figure, record, cfg, omitted)
            return Ok(figure)
    except (OSError, ValueError, RuntimeError, ZeroDivisionError) as exc:
        if figure is not None:
            figure.clf()
            import matplotlib.pyplot as plt

            plt.close(figure)
        return Err(f"cannot render figure {record.identifier}: {exc}")


def render_training_figures(
    figures: tuple[TrainingFigure, ...], cfg: PlotConfig
) -> Result[RenderedDatasetFigures, str]:
    """Export every frozen training recipe into per-format bundle bytes.

    Files land under ``figures/training/{identifier}.{format}`` so they
    cannot collide with dataset figures. A recipe whose reason is
    recorded - or whose log-domain masking removes every point - renders
    as a labelled placeholder and is indexed ``UNAVAILABLE``; all other
    figures are ``AVAILABLE``.
    """
    files: list[BundleFile] = []
    outputs: list[AvailabilityRecord] = []
    for record in figures:
        rendered = training_figure(record, cfg)
        if isinstance(rendered, Err):
            return rendered
        exported = export_figure(rendered.value, "training/" + record.identifier, cfg)
        if isinstance(exported, Err):
            return exported
        files.extend(exported.value)
        reason = _effective_reason(record)
        outputs.append(
            AvailabilityRecord(
                name="training_figure:" + record.identifier,
                status="UNAVAILABLE" if reason is not None else "AVAILABLE",
                reason=reason,
            )
        )
    return Ok(RenderedDatasetFigures(files=tuple(files), outputs=tuple(outputs)))
