"""Bar charts for the band matrix.

Contains:
  - write_band_bars: one bar per subset name.
  - write_metric_bars: one chart for one metric.
  - write_delta_bars: score minus the 12-band baseline.
  - write_loss_curve: linear-epoch loss with a labeled selected epoch.
  - write_pr_curve: precision-recall curve.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def write_band_bars(names: Sequence[str], scores: Sequence[float], path: Path) -> None:
    """Write a horizontal bar chart.

    Args:
        names: Subset names, one per bar.
        scores: Score aligned with ``names``.
        path: PNG destination. Parent directories are created.

    Raises:
        ValueError: If the two sequences differ in length or are empty.
    """
    if not names or len(names) != len(scores):
        raise ValueError(
            f"need one score per name; got {len(names)} names and {len(scores)} scores"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    figure, axis = plt.subplots(figsize=(8, max(2.0, 0.35 * len(names))))
    axis.barh(list(names), list(scores))
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("score")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def write_metric_bars(
    title: str,
    names: Sequence[str],
    scores: Sequence[float],
    path: Path,
) -> None:
    """Write one horizontal bar chart for a single metric.

    Args:
        title: Axis label and figure title.
        names: Subset names, one per bar.
        scores: Metric values aligned with ``names``.
        path: PNG destination. Parent directories are created.

    Raises:
        ValueError: If the two sequences differ in length or are empty.
    """
    if not names or len(names) != len(scores):
        raise ValueError(
            f"need one score per name; got {len(names)} names and {len(scores)} scores"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    figure, axis = plt.subplots(figsize=(8, max(2.0, 0.35 * len(names))))
    axis.barh(list(names), list(scores))
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel(title)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def write_delta_bars(
    title: str,
    names: Sequence[str],
    deltas: Sequence[float],
    path: Path,
) -> None:
    """Write bars of score minus the 12-band baseline, largest drop at the top.

    Args:
        title: Axis label and figure title.
        names: Subset names, one per bar.
        deltas: Difference from the baseline, aligned with ``names``.
        path: PNG destination. Parent directories are created.

    Raises:
        ValueError: If the two sequences differ in length or are empty.
    """
    if not names or len(names) != len(deltas):
        raise ValueError(
            f"need one delta per name; got {len(names)} names and {len(deltas)} deltas"
        )
    order = sorted(range(len(names)), key=lambda index: deltas[index])
    ordered_names = [names[index] for index in order]
    ordered_deltas = [deltas[index] for index in order]
    path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    figure, axis = plt.subplots(figsize=(8, max(2.0, 0.35 * len(names))))
    axis.barh(ordered_names, ordered_deltas)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_xlabel(title)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def write_pr_curve(
    recall: Sequence[float],
    precision: Sequence[float],
    path: Path,
    *,
    title: str,
) -> None:
    """Write one precision-recall curve.

    Args:
        recall: Recall coordinates, including the origin.
        precision: Precision coordinates aligned with ``recall``.
        path: PNG destination. Parent directories are created.
        title: Figure title.

    Raises:
        ValueError: If the sequences differ in length or are empty.
    """
    if not recall or len(recall) != len(precision):
        raise ValueError("need one precision per recall")
    path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(list(recall), list(precision))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def write_loss_curve(
    epochs: Sequence[int],
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    test_loss: Sequence[float],
    path: Path,
    *,
    selected_epoch: int,
) -> None:
    """Write loss against a linear epoch axis, with loss on a log scale.

    Args:
        epochs: One-based epoch index for each point.
        train_loss: Training loss aligned with ``epochs``.
        val_loss: Validation loss aligned with ``epochs``.
        test_loss: Test loss aligned with ``epochs``.
        path: PNG destination. Parent directories are created.
        selected_epoch: One-based epoch whose weights are reported.

    Raises:
        ValueError: If the sequences differ in length or are empty, or a loss
            is not positive.
    """
    series = (train_loss, val_loss, test_loss)
    if not epochs or any(len(item) != len(epochs) for item in series):
        raise ValueError("need one train, validation, and test loss per epoch")
    for values in series:
        if any(value <= 0.0 for value in values):
            raise ValueError("loss values must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.semilogy(list(epochs), list(train_loss), label="train")
    axis.semilogy(list(epochs), list(val_loss), label="validation")
    axis.semilogy(list(epochs), list(test_loss), label="test")
    axis.axvline(selected_epoch, color="black", linestyle="--", label=f"epoch {selected_epoch}")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
