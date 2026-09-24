"""Bar charts for the band matrix.

Contains:
  - write_band_bars: one bar per subset name.
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
