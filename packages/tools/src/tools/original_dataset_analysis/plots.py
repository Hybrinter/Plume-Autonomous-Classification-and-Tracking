"""Bar charts for the band matrix.

Contains:
  - write_band_bars: one bar per subset name.
  - write_metric_bars: one chart for one metric.
  - write_loss_curve: log-log train, validation, and test loss.
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


def write_loss_curve(
    epochs: Sequence[int],
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    test_loss: Sequence[float],
    path: Path,
    *,
    selected_epoch: int,
) -> None:
    """Write a log-log loss chart with a marker on the selected epoch.

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
            raise ValueError("log-log loss values must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.loglog(list(epochs), list(train_loss), label="train")
    axis.loglog(list(epochs), list(val_loss), label="validation")
    axis.loglog(list(epochs), list(test_loss), label="test")
    axis.axvline(selected_epoch, color="black", linestyle="--", label="selected")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
