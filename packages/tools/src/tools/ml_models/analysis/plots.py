"""Headless matplotlib figures for runs, chips, and the band study.

Selecting the Agg backend on import keeps figure generation display-free.
This module does not import tools.analysis.

Contains:
  - LabeledFigure: named Figure ready to save.
  - history_figures: train/val loss and metric curves from history.csv.
  - overlay_figures: prediction versus gold overlays from predictions.npz.
  - failure_figures: lowest-scoring preview samples.
  - save_figures: PNG emission.
  - write_band_bars, write_metric_bars, write_delta_bars: study bar charts.
  - write_pr_curve, write_loss_curve, write_learning_rate, write_gsd_lines.
  - write_canvas_preview, write_score_histogram, write_reliability.
  - write_hit_rate_by_placement, write_empty_fpr, write_logit_margin.
  - write_blob_area_histogram.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import csv
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from tools.ml_models.train.metrics import sigmoid  # noqa: E402

_FIGSIZE = (9.0, 4.5)
_DPI = 110


@dataclass(frozen=True, slots=True)
class LabeledFigure:
    """A named, titled matplotlib Figure ready to save into a run directory."""

    name: str
    title: str
    figure: Figure


def save_figures(figures: list[LabeledFigure], out_dir: Path) -> list[Path]:
    """Save each figure as a PNG into out_dir and close it.

    Args:
        figures: Figures to write.
        out_dir: Destination directory.

    Returns:
        list[Path]: Written PNG paths in input order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for labeled in figures:
        path = out_dir / f"{labeled.name}.png"
        labeled.figure.savefig(path, bbox_inches="tight")
        plt.close(labeled.figure)
        written.append(path)
    return written


def history_figures(history_csv: str | Path) -> list[LabeledFigure]:
    """Build train/val curves from history.csv.

    Args:
        history_csv: Epoch history written by train.

    Returns:
        list[LabeledFigure]: One figure per numeric metric column.
    """
    path = Path(history_csv)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    skip = {"epoch", "split"}
    metrics = [key for key in rows[0] if key not in skip]
    figures: list[LabeledFigure] = []
    for metric in metrics:
        figure, axes = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
        drew = False
        for split in ("train", "val"):
            xs = [int(row["epoch"]) for row in rows if row["split"] == split]
            ys = [float(row[metric]) for row in rows if row["split"] == split]
            if not xs:
                continue
            axes.plot(xs, ys, marker=".", label=split)
            drew = True
        if not drew:
            plt.close(figure)
            continue
        axes.set_title(metric)
        axes.set_xlabel("epoch")
        axes.set_ylabel(metric)
        axes.grid(visible=True, alpha=0.3)
        axes.legend(fontsize="small")
        figure.tight_layout()
        figures.append(LabeledFigure(name=f"curve_{metric}", title=metric, figure=figure))
    return figures


def _rgb(image: np.ndarray) -> np.ndarray:
    """Return an (H, W, 3) preview from a (C, H, W) tensor."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected (C, H, W); got {arr.shape}")
    channels = arr[:3]
    if channels.shape[0] == 1:
        channels = np.repeat(channels, 3, axis=0)
    elif channels.shape[0] == 2:
        pad = np.zeros((1, arr.shape[1], arr.shape[2]), dtype=np.float32)
        channels = np.concatenate([channels, pad], axis=0)
    rgb = np.transpose(channels[:3], (1, 2, 0))
    return np.clip(rgb, 0.0, 1.0)


def overlay_figures(predictions_npz: str | Path, limit: int = 4) -> list[LabeledFigure]:
    """Build RGB / gold / prediction panels from predictions.npz.

    Args:
        predictions_npz: Array archive written by evaluate.
        limit: Max samples to draw.

    Returns:
        list[LabeledFigure]: One figure, or empty when the archive is missing.
    """
    path = Path(predictions_npz)
    if not path.is_file():
        return []
    payload = np.load(path, allow_pickle=False)
    images = np.asarray(payload["images"])
    targets = np.asarray(payload["targets"])
    logits = np.asarray(payload["logits"])
    kind = str(payload["kind"])
    n = min(int(images.shape[0]), int(limit))
    if n == 0:
        return []
    cols = 3 if kind == "segmentor" else 2
    figure, axes = plt.subplots(n, cols, figsize=(3.2 * cols, 3.0 * n), dpi=_DPI)
    axes_arr = np.atleast_2d(axes)
    probs = np.asarray(sigmoid(logits).detach().cpu().numpy())
    for i in range(n):
        axes_arr[i, 0].imshow(_rgb(images[i]))
        axes_arr[i, 0].set_title("input")
        axes_arr[i, 0].axis("off")
        if kind == "segmentor":
            gold = targets[i, 0] if targets.ndim == 4 else targets[i]
            pred = probs[i, 0] if probs.ndim == 4 else probs[i]
            axes_arr[i, 1].imshow(gold, vmin=0.0, vmax=1.0, cmap="gray")
            axes_arr[i, 1].set_title("gold")
            axes_arr[i, 1].axis("off")
            axes_arr[i, 2].imshow(pred, vmin=0.0, vmax=1.0, cmap="gray")
            axes_arr[i, 2].set_title("pred")
            axes_arr[i, 2].axis("off")
        else:
            label = float(targets.reshape(targets.shape[0], -1)[i, 0])
            logit = float(logits.reshape(logits.shape[0], -1)[i, 0])
            axes_arr[i, 1].imshow(_rgb(images[i]))
            axes_arr[i, 1].set_title(f"label={label:.0f} logit={logit:.2f}")
            axes_arr[i, 1].axis("off")
    figure.tight_layout()
    return [LabeledFigure(name="overlays", title="prediction overlays", figure=figure)]


def failure_figures(predictions_npz: str | Path, limit: int = 4) -> list[LabeledFigure]:
    """Build a gallery of the lowest-scoring preview samples.

    Args:
        predictions_npz: Array archive written by evaluate.
        limit: Max samples to draw.

    Returns:
        list[LabeledFigure]: Empty when every preview score is 1.0 or the file
        is missing.
    """
    path = Path(predictions_npz)
    if not path.is_file():
        return []
    payload = np.load(path, allow_pickle=False)
    scores = np.asarray(payload["scores"], dtype=np.float64)
    if scores.size == 0 or float(np.min(scores)) >= 1.0:
        return []
    order = np.argsort(scores)[: int(limit)]
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
        tmp = Path(handle.name)
    np.savez(
        tmp,
        images=np.asarray(payload["images"])[order],
        targets=np.asarray(payload["targets"])[order],
        logits=np.asarray(payload["logits"])[order],
        scores=scores[order],
        kind=payload["kind"],
    )
    try:
        figures = overlay_figures(tmp, limit=limit)
    finally:
        tmp.unlink(missing_ok=True)
    if not figures:
        return []
    labeled = figures[0]
    labeled.figure.suptitle("lowest-scoring samples")
    return [LabeledFigure(name="failures", title="failure gallery", figure=labeled.figure)]


def _save(figure: Figure, path: Path) -> Path:
    """Write one PNG and close the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def _require_pairs(left: Sequence[object], right: Sequence[object], noun: str) -> None:
    """Raise ValueError when the two sequences are empty or differ in length."""
    if not left or len(left) != len(right):
        raise ValueError(f"need one {noun} per name; got {len(left)} and {len(right)}")


def write_band_bars(names: Sequence[str], scores: Sequence[float], path: Path) -> Path:
    """Write a horizontal bar chart.

    Args:
        names: Subset names, one per bar.
        scores: Score aligned with ``names``.
        path: PNG destination. Parent directories are created.

    Returns:
        Path: The written PNG.

    Raises:
        ValueError: If the two sequences differ in length or are empty.
    """
    _require_pairs(names, scores, "score")
    figure, axis = plt.subplots(figsize=(8, max(2.0, 0.35 * len(names))))
    axis.barh(list(names), list(scores))
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("score")
    figure.tight_layout()
    return _save(figure, path)


def write_metric_bars(
    title: str,
    names: Sequence[str],
    scores: Sequence[float],
    path: Path,
) -> Path:
    """Write one horizontal bar chart for a single metric.

    Args:
        title: Axis label and figure title.
        names: Subset names, one per bar.
        scores: Metric values aligned with ``names``.
        path: PNG destination. Parent directories are created.

    Returns:
        Path: The written PNG.

    Raises:
        ValueError: If the two sequences differ in length or are empty.
    """
    _require_pairs(names, scores, "score")
    figure, axis = plt.subplots(figsize=(8, max(2.0, 0.35 * len(names))))
    axis.barh(list(names), list(scores))
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel(title)
    axis.set_title(title)
    figure.tight_layout()
    return _save(figure, path)


def write_delta_bars(
    title: str,
    names: Sequence[str],
    deltas: Sequence[float],
    path: Path,
) -> Path:
    """Write bars of score minus the 12-band baseline, largest drop at the top.

    Args:
        title: Axis label and figure title.
        names: Subset names, one per bar.
        deltas: Difference from the baseline, aligned with ``names``.
        path: PNG destination. Parent directories are created.

    Returns:
        Path: The written PNG.

    Raises:
        ValueError: If the two sequences differ in length or are empty.
    """
    _require_pairs(names, deltas, "delta")
    order = sorted(range(len(names)), key=lambda index: deltas[index])
    ordered_names = [names[index] for index in order]
    ordered_deltas = [deltas[index] for index in order]
    figure, axis = plt.subplots(figsize=(8, max(2.0, 0.35 * len(names))))
    axis.barh(ordered_names, ordered_deltas)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_xlabel(title)
    axis.set_title(title)
    figure.tight_layout()
    return _save(figure, path)


def write_pr_curve(
    recall: Sequence[float],
    precision: Sequence[float],
    path: Path,
    *,
    title: str,
) -> Path:
    """Write one precision-recall curve.

    Args:
        recall: Recall coordinates, including the origin.
        precision: Precision coordinates aligned with ``recall``.
        path: PNG destination. Parent directories are created.
        title: Figure title.

    Returns:
        Path: The written PNG.

    Raises:
        ValueError: If the sequences differ in length or are empty.
    """
    if not recall or len(recall) != len(precision):
        raise ValueError("need one precision per recall")
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(list(recall), list(precision))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_title(title)
    figure.tight_layout()
    return _save(figure, path)


def write_loss_curve(
    epochs: Sequence[int],
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    test_loss: Sequence[float],
    path: Path,
    *,
    selected_epoch: int,
) -> Path:
    """Write loss against a linear epoch axis, with loss on a log scale.

    Args:
        epochs: One-based epoch index for each point.
        train_loss: Training loss aligned with ``epochs``.
        val_loss: Validation loss aligned with ``epochs``.
        test_loss: Test loss aligned with ``epochs``.
        path: PNG destination. Parent directories are created.
        selected_epoch: One-based epoch whose weights are reported.

    Returns:
        Path: The written PNG.

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
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.semilogy(list(epochs), list(train_loss), label="train")
    axis.semilogy(list(epochs), list(val_loss), label="validation")
    axis.semilogy(list(epochs), list(test_loss), label="test")
    axis.axvline(selected_epoch, color="black", linestyle="--", label=f"epoch {selected_epoch}")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.legend()
    figure.tight_layout()
    return _save(figure, path)


def write_learning_rate(
    epochs: Sequence[int],
    learning_rate: Sequence[float],
    path: Path,
) -> Path:
    """Write learning rate against epoch.

    Args:
        epochs: One-based epoch index for each point.
        learning_rate: Learning rate aligned with ``epochs``.
        path: PNG destination. Parent directories are created.

    Returns:
        Path: The written PNG.

    Raises:
        ValueError: If the sequences differ in length or are empty.
    """
    if not epochs or len(epochs) != len(learning_rate):
        raise ValueError("need one learning rate per epoch")
    figure, axis = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    axis.plot(list(epochs), list(learning_rate), marker=".")
    axis.set_xlabel("epoch")
    axis.set_ylabel("learning_rate")
    axis.set_title("learning_rate")
    axis.grid(visible=True, alpha=0.3)
    figure.tight_layout()
    return _save(figure, path)


def write_gsd_lines(
    scores: Mapping[tuple[str, str, int], float],
    path: Path,
) -> Path:
    """Write one score per task against ground-sample distance.

    Args:
        scores: Metric keyed by ``(task, subset, side_px)``. The classify
            series is ``pr_auc``. The segment series is ``native_dice``, Dice
            after the coarse map is expanded onto the 120 px mask. Coarse Dice
            is not drawn on that axis.
        path: PNG destination. Parent directories are created.

    Returns:
        Path: The written PNG.

    Raises:
        ValueError: If either head is missing a 12-band or RGB point.
    """
    from tools.ml_models.data.grid import gsd_m

    sides = (120, 80, 60, 40, 30)
    series = (
        ("classify", "pr_auc"),
        ("segment", "native_dice"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    metres = [gsd_m(side) for side in sides]
    for axis, (task, ylabel) in zip(axes, series, strict=True):
        for subset in ("s2_12", "rgb"):
            values = [scores.get((task, subset, side)) for side in sides]
            if any(value is None for value in values):
                plt.close(figure)
                raise ValueError(f"missing {task} {subset} ground-sample score")
            axis.plot(
                metres,
                [float(value) for value in values if value is not None],
                marker="o",
                label=subset,
            )
        axis.set_xlabel("ground sample distance (m)")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 1.0)
        axis.legend()
    figure.tight_layout()
    return _save(figure, path)


def _preview_rgb(image: np.ndarray) -> np.ndarray:
    """Return an ``(H, W, 3)`` preview from CHW or HWC."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected rank-3 image; got {arr.shape}")
    if arr.shape[-1] in {1, 3, 4} and arr.shape[0] > 4:
        rgb = arr[..., :3]
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        return np.clip(rgb, 0.0, 1.0)
    return _rgb(arr)


def write_canvas_preview(image: np.ndarray, mask: np.ndarray, path: Path) -> Path:
    """Draw one canvas image beside its mask.

    Args:
        image: Float array ``(C, H, W)`` or ``(H, W, 3)``.
        mask: Float mask ``(H, W)`` or ``(1, H, W)``. Intermediate values on
            the border stay visible.
        path: PNG destination.

    Returns:
        Path: The written PNG.

    Raises:
        ValueError: If the image rank is not 3.
    """
    rgb = _preview_rgb(image)
    plane = np.asarray(mask, dtype=np.float32)
    if plane.ndim == 3:
        plane = plane[0]
    figure, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), dpi=_DPI)
    axes[0].imshow(np.clip(rgb, 0.0, 1.0))
    axes[0].set_title("canvas")
    axes[0].axis("off")
    axes[1].imshow(plane, vmin=0.0, vmax=1.0, cmap="gray")
    axes[1].set_title("mask")
    axes[1].axis("off")
    figure.tight_layout()
    return _save(figure, path)


def write_score_histogram(scores: Sequence[float], path: Path) -> Path:
    """Write a histogram of classifier scores.

    Args:
        scores: One score per sample.
        path: PNG destination.

    Returns:
        Path: The written PNG.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    if not scores:
        raise ValueError("scores must be non-empty")
    figure, axis = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    axis.hist(list(scores), bins=min(10, len(scores)), range=(0.0, 1.0))
    axis.set_xlabel("score")
    axis.set_ylabel("count")
    axis.set_title("score histogram")
    figure.tight_layout()
    return _save(figure, path)


def write_reliability(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    path: Path,
    *,
    bins: int = 8,
) -> Path:
    """Write mean outcome against mean probability in equal-width bins.

    Args:
        probabilities: Predicted probabilities in ``[0, 1]``.
        outcomes: Matching binary outcomes.
        path: PNG destination.
        bins: Equal-width bin count on ``[0, 1]``.

    Returns:
        Path: The written PNG.

    Raises:
        ValueError: If the sequences differ in length, are empty, or ``bins``
            is below 1.
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1; got {bins}")
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("need one outcome per probability")
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(outcomes, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers: list[float] = []
    means: list[float] = []
    for index in range(bins):
        lo = edges[index]
        hi = edges[index + 1]
        if index == bins - 1:
            chosen = (probs >= lo) & (probs <= hi)
        else:
            chosen = (probs >= lo) & (probs < hi)
        if not np.any(chosen):
            continue
        centers.append(float(np.mean(probs[chosen])))
        means.append(float(np.mean(labels[chosen])))
    figure, axis = plt.subplots(figsize=(5.5, 5.0), dpi=_DPI)
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=1)
    if centers:
        axis.plot(centers, means, marker="o")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("mean probability")
    axis.set_ylabel("mean outcome")
    axis.set_title("reliability")
    figure.tight_layout()
    return _save(figure, path)


def write_hit_rate_by_placement(rates: Mapping[str, float], path: Path) -> Path:
    """Write hit rate bars for center, corner, and edge.

    Args:
        rates: Hit rate keyed by placement. Missing keys are drawn as 0.
        path: PNG destination.

    Returns:
        Path: The written PNG.
    """
    names = ("center", "corner", "edge")
    values = [float(rates.get(name, 0.0)) for name in names]
    figure, axis = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    axis.bar(list(names), values)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("hit rate")
    axis.set_title("hit rate by placement")
    figure.tight_layout()
    return _save(figure, path)


def write_empty_fpr(thresholds: Sequence[float], rates: Sequence[float], path: Path) -> Path:
    """Write empty-frame false-positive rate against threshold.

    Args:
        thresholds: Decision thresholds.
        rates: False-positive rate aligned with ``thresholds``.
        path: PNG destination.

    Returns:
        Path: The written PNG.

    Raises:
        ValueError: If the sequences differ in length or are empty.
    """
    if not thresholds or len(thresholds) != len(rates):
        raise ValueError("need one false-positive rate per threshold")
    figure, axis = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    axis.plot(list(thresholds), list(rates), marker="o")
    axis.set_xlabel("threshold")
    axis.set_ylabel("empty-frame false-positive rate")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("empty-frame false-positive rate")
    figure.tight_layout()
    return _save(figure, path)


def write_logit_margin(
    chip_logits: Sequence[float],
    frame_logits: Sequence[float],
    path: Path,
) -> Path:
    """Write chip max logit against full-frame max logit for the same plume.

    Args:
        chip_logits: Max logit on the source chip.
        frame_logits: Max logit on the full frame, aligned with ``chip_logits``.
        path: PNG destination.

    Returns:
        Path: The written PNG.

    Raises:
        ValueError: If the sequences differ in length or are empty.
    """
    if not chip_logits or len(chip_logits) != len(frame_logits):
        raise ValueError("need one frame logit per chip logit")
    figure, axis = plt.subplots(figsize=(5.5, 5.0), dpi=_DPI)
    axis.scatter(list(chip_logits), list(frame_logits))
    lo = min(*chip_logits, *frame_logits)
    hi = max(*chip_logits, *frame_logits)
    axis.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("chip max logit")
    axis.set_ylabel("full-frame max logit")
    axis.set_title("chip versus frame logit")
    figure.tight_layout()
    return _save(figure, path)


def write_blob_area_histogram(areas: Sequence[float], path: Path) -> Path:
    """Write a histogram of blob areas in pixels.

    Args:
        areas: One area per blob.
        path: PNG destination.

    Returns:
        Path: The written PNG.

    Raises:
        ValueError: If ``areas`` is empty.
    """
    if not areas:
        raise ValueError("areas must be non-empty")
    figure, axis = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    axis.hist(list(areas), bins=min(10, len(areas)))
    axis.set_xlabel("blob area (px)")
    axis.set_ylabel("count")
    axis.set_title("blob area")
    figure.tight_layout()
    return _save(figure, path)
