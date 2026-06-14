"""Shared matplotlib primitives for the per-group figure builders (headless, deterministic).

Reusable figure constructors over a group's wide frame: step-indexed numeric line panels, a
categorical-state timeline (labels mapped to ordinals), per-step stacked count areas, cumulative
count lines, and a value-versus-limit overlay. Each returns a LabeledFigure (a named Figure) or
None when the requested columns are absent or carry no finite data, so a builder can drop empty
panels with a simple filter. The Agg backend is selected on import so rendering never needs a
display. The shared frames are deterministic, so the figures are too.

Contains:
  - LabeledFigure: a named, titled matplotlib Figure.
  - line_panel / categorical_timeline / stacked_counts / cumulative_lines / value_with_limit:
    figure constructors over a wide frame.
  - present_numeric / save_figures: column filtering + PNG emission helpers.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass
from pathlib import Path

# third-party
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 (must follow the Agg backend selection)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

_FIGSIZE = (9.0, 4.5)
_DPI = 110


@dataclass(frozen=True, slots=True)
class LabeledFigure:
    """A named, titled matplotlib Figure ready to save into a report bundle."""

    name: str
    title: str
    figure: Figure


def _label(column: str) -> str:
    """Strip the leading group prefix from a signal name for a compact legend label."""
    return column.split(".", 1)[1] if "." in column else column


def present_numeric(wide: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return the subset of columns present in wide that carry at least one finite value."""
    kept: list[str] = []
    for column in columns:
        if column in wide.columns:
            series = pd.to_numeric(wide[column], errors="coerce")
            if bool(np.isfinite(series.to_numpy(dtype="float64")).any()):
                kept.append(column)
    return kept


def line_panel(
    wide: pd.DataFrame,
    columns: list[str],
    *,
    name: str,
    title: str,
    ylabel: str,
) -> LabeledFigure | None:
    """Build a line panel of one or more numeric columns versus step.

    Args:
        wide: the group's wide frame (step-indexed).
        columns: the numeric signal columns to plot (absent/empty ones are skipped).
        name: the figure file stem.
        title: the figure title.
        ylabel: the y-axis label.

    Returns:
        A LabeledFigure, or None if none of the columns had finite data.
    """
    usable = present_numeric(wide, columns)
    if not usable:
        return None
    steps = wide.index.to_numpy()
    figure, axes = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    for column in usable:
        axes.plot(steps, wide[column].to_numpy(dtype="float64"), label=_label(column), marker=".")
    axes.set_title(title)
    axes.set_xlabel("step")
    axes.set_ylabel(ylabel)
    axes.grid(visible=True, alpha=0.3)
    if len(usable) > 1:
        axes.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    return LabeledFigure(name, title, figure)


def categorical_timeline(
    wide: pd.DataFrame, column: str, *, name: str, title: str
) -> LabeledFigure | None:
    """Build a step timeline of a categorical column (labels mapped to ordinal y values).

    Args:
        wide: the group's wide frame.
        column: the categorical signal column.
        name: the figure file stem.
        title: the figure title.

    Returns:
        A LabeledFigure, or None if the column is absent or empty.
    """
    if column not in wide.columns:
        return None
    labels = wide[column].astype("string").fillna("").to_numpy()
    categories = [value for value in dict.fromkeys(labels.tolist()) if value != ""]
    if not categories:
        return None
    ordinal = {value: position for position, value in enumerate(categories)}
    ys = [ordinal.get(str(value), np.nan) for value in labels]
    steps = wide.index.to_numpy()
    figure, axes = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    axes.step(steps, ys, where="post", marker="o")
    axes.set_yticks(range(len(categories)))
    axes.set_yticklabels(categories)
    axes.set_title(title)
    axes.set_xlabel("step")
    axes.set_ylabel("state")
    axes.grid(visible=True, axis="x", alpha=0.3)
    figure.tight_layout()
    return LabeledFigure(name, title, figure)


def stacked_counts(
    wide: pd.DataFrame,
    columns: list[str],
    *,
    name: str,
    title: str,
    ylabel: str = "count / step",
) -> LabeledFigure | None:
    """Build a stacked area of per-step count columns versus step."""
    usable = present_numeric(wide, columns)
    usable = [c for c in usable if float(wide[c].to_numpy(dtype="float64").sum()) > 0.0]
    if not usable:
        return None
    steps = wide.index.to_numpy()
    stacks = [np.nan_to_num(wide[column].to_numpy(dtype="float64")) for column in usable]
    figure, axes = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    axes.stackplot(steps, *stacks, labels=[_label(column) for column in usable])
    axes.set_title(title)
    axes.set_xlabel("step")
    axes.set_ylabel(ylabel)
    axes.legend(fontsize="small", ncol=2, loc="upper left")
    figure.tight_layout()
    return LabeledFigure(name, title, figure)


def cumulative_lines(
    wide: pd.DataFrame, columns: list[str], *, name: str, title: str
) -> LabeledFigure | None:
    """Build a line panel of the cumulative derivations of the given count columns."""
    cumulative = [f"{column}.cumulative" for column in columns]
    return line_panel(wide, cumulative, name=name, title=title, ylabel="cumulative count")


def value_with_limit(
    wide: pd.DataFrame,
    value_column: str,
    limit_column: str,
    *,
    name: str,
    title: str,
    ylabel: str,
) -> LabeledFigure | None:
    """Build a value-versus-limit overlay (the value series with its threshold as a dashed line)."""
    if value_column not in wide.columns:
        return None
    steps = wide.index.to_numpy()
    values = pd.to_numeric(wide[value_column], errors="coerce").to_numpy(dtype="float64")
    if not bool(np.isfinite(values).any()):
        return None
    figure, axes = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    axes.plot(steps, values, label=_label(value_column), marker=".", color="tab:blue")
    if limit_column in wide.columns:
        limit = pd.to_numeric(wide[limit_column], errors="coerce").to_numpy(dtype="float64")
        if bool(np.isfinite(limit).any()):
            axes.plot(steps, limit, label=_label(limit_column), linestyle="--", color="tab:red")
    axes.set_title(title)
    axes.set_xlabel("step")
    axes.set_ylabel(ylabel)
    axes.grid(visible=True, alpha=0.3)
    axes.legend(fontsize="small")
    figure.tight_layout()
    return LabeledFigure(name, title, figure)


def save_figures(figures: list[LabeledFigure], out_dir: Path) -> list[Path]:
    """Save each figure as a PNG into out_dir and close it; return the written paths.

    Args:
        figures: the figures to write.
        out_dir: the destination directory (created if needed).

    Returns:
        The list of written PNG paths, in input order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for labeled in figures:
        path = out_dir / f"{labeled.name}.png"
        labeled.figure.savefig(path, bbox_inches="tight")
        plt.close(labeled.figure)
        written.append(path)
    return written
