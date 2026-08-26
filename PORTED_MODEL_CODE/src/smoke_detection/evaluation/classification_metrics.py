"""Classification eval plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def plot_confusion_matrix(tp: int, tn: int, fp: int, fn: int, out_path: str | Path) -> None:
    """Save a 2x2 confusion-matrix heatmap."""
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True Neg", "True Pos"])
    ax.set_title("Confusion Matrix")
    for r in range(2):
        for c in range(2):
            ax.text(
                c,
                r,
                str(cm[r, c]),
                ha="center",
                va="center",
                fontsize=14,
                color="white" if cm[r, c] > cm.max() / 2 else "black",
            )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(scores: list[float], labels: list[int], out_path: str | Path) -> None:
    """Save an ROC-curve plot; annotates AUC."""
    fpr, tpr, _ = roc_curve(labels, scores)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
