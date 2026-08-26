"""Segmentation eval plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_iou_distribution(ious: list[float], out_path: str | Path) -> None:
    """Save an IoU distribution histogram."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ious, bins=20, edgecolor="black")
    mean = float(np.mean(ious)) if ious else 0.0
    ax.axvline(mean, color="red", linestyle="--", label=f"Mean = {mean:.3f}")
    ax.set_xlabel("IoU Score")
    ax.set_ylabel("Count")
    ax.set_title(f"IoU Distribution (n={len(ious)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_area_ratio_distribution(ratios: list[float], out_path: str | Path) -> None:
    """Save a pred/true area-ratio histogram."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ratios, bins=30, edgecolor="black")
    ax.axvline(1.0, color="red", linestyle="--", label="Reference = 1.0")
    mean = float(np.mean(ratios)) if ratios else 0.0
    ax.axvline(mean, color="orange", linestyle="--", label=f"Mean = {mean:.3f}")
    ax.set_xlabel("Area Ratio (pred / true)")
    ax.set_ylabel("Count")
    ax.set_title(f"Area Ratio Distribution (n={len(ratios)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
