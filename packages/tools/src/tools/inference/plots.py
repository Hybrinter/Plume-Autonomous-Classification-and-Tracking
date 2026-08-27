"""Headless matplotlib figures for inference run directories.

Selecting the Agg backend on import keeps figure generation display-free.
This module does not import tools.analysis.

Contains:
  - LabeledFigure: named Figure ready to save.
  - history_figures: train/val loss and metric curves from history.csv.
  - overlay_figures: prediction versus gold overlays from predictions.npz.
  - failure_figures: lowest-scoring preview samples.
  - save_figures: PNG emission.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from tools.inference.metrics import sigmoid  # noqa: E402

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
    probs = sigmoid(logits)
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
