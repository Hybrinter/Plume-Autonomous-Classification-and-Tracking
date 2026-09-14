"""Shared matplotlib style for design-study figures.

Contains:
  - Colour constants used by study figure modules.
  - apply: set rcParams for white-background report figures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

C_BLUE = "#0077BB"
C_ORANGE = "#EE7733"
C_TEAL = "#009988"
C_RED = "#CC3311"
C_SKY = "#33BBEE"
C_INK = "#222222"


def apply() -> None:
    """Set matplotlib rcParams for white-background report figures.

    Returns:
        None. Mutates the global rcParams.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "axes.edgecolor": C_INK,
            "text.color": C_INK,
            "axes.labelcolor": C_INK,
            "xtick.color": C_INK,
            "ytick.color": C_INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
