"""Figures for the VIS/SWIR fusion trade.

Contains:
  - write_seed_plot: one score per seed for each option, with the seed mean.
  - write_threshold_plot: test F1 per look-angle bin for each threshold method.

Both figures are static PNGs for a results page, so the numbers they show are
also written to the JSON outputs that feed them. The threshold plot gives each
method its own marker shape as well as its own color, so it reads without color.

Satisfies: REQ-AIML-COMP-001.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from tools.ml_models.fusion.thresholds import ALL_BINS, ThresholdRow  # noqa: E402

_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
_INK = "#1f2937"
_MUTED = "#6b7280"
_GRID = "#e5e7eb"


def _style(ax: Axes) -> None:
    """Apply recessive axes and a horizontal grid.

    Args:
        ax: Axes edited in place.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelcolor=_INK)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def write_seed_plot(
    scores: Mapping[str, Sequence[float]], metric_label: str, title: str, path: Path
) -> None:
    """Plot each option's per-seed scores as dots and its mean as a bar tick.

    Args:
        scores: Option label to one score per seed, in display order.
        metric_label: Y-axis label, such as ``Test average precision``.
        title: Figure title.
        path: Output PNG; the parent directory is created.

    Raises:
        ValueError: If ``scores`` is empty or an option has no score.
    """
    if not scores or any(len(v) == 0 for v in scores.values()):
        raise ValueError("every option needs at least one score")
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    _style(ax)
    for x, values in enumerate(scores.values()):
        mean = sum(values) / len(values)
        ax.hlines(mean, x - 0.25, x + 0.25, color=_INK, linewidth=2)
        ax.scatter(
            [x] * len(values),
            values,
            s=64,
            color=_SERIES[0],
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        ax.annotate(
            f"{mean:.3f}",
            (x + 0.27, mean),
            va="center",
            fontsize=9,
            color=_INK,
        )
    ax.set_xticks(range(len(scores)), list(scores))
    ax.set_xlim(-0.6, len(scores) - 0.3)
    ax.set_ylabel(metric_label, color=_INK)
    ax.set_title(title, color=_INK, loc="left", fontsize=11)
    ax.text(
        0.0,
        -0.18,
        "Dots: one training seed each. Bar: mean over seeds.",
        transform=ax.transAxes,
        fontsize=8,
        color=_MUTED,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_threshold_plot(rows: Sequence[ThresholdRow], title: str, path: Path) -> None:
    """Plot test F1 per look-angle bin at the fixed, global, and per-bin thresholds.

    Args:
        rows: Rows from ``score_thresholds``; the ``all`` row goes in the title.
        title: Figure title prefix.
        path: Output PNG; the parent directory is created.

    Raises:
        ValueError: If ``rows`` has no per-bin row.
    """
    per_bin = [r for r in rows if r.bin_id != ALL_BINS]
    if not per_bin:
        raise ValueError("threshold plot needs at least one look-angle bin")
    total = next((r for r in rows if r.bin_id == ALL_BINS), None)
    labels = [(r.bin_id.removeprefix("el").lstrip("0") or "0") + " deg" for r in per_bin]
    xs = list(range(len(per_bin)))
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    _style(ax)
    series = (
        ("Fixed 0.5", [r.f1_fixed for r in per_bin], "o"),
        ("One global threshold", [r.f1_global for r in per_bin], "s"),
        ("Threshold per look angle", [r.f1_per_bin for r in per_bin], "^"),
    )
    for (name, values, marker), color in zip(series, _SERIES, strict=True):
        ax.plot(xs, values, color=color, linewidth=2, marker=marker, markersize=8, label=name)
    ax.set_xticks(xs, labels)
    ax.set_xlabel("Look angle (elevation bin)", color=_INK)
    ax.set_ylabel("Test F1", color=_INK)
    ax.set_ylim(0.0, 1.0)
    if total is not None:
        title = (
            f"{title}: F1 over all tiles {total.f1_fixed:.3f} (fixed) / "
            f"{total.f1_global:.3f} (global) / {total.f1_per_bin:.3f} (per angle)"
        )
    ax.set_title(title, color=_INK, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=_INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
