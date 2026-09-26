"""Training objectives for the classifier and the segmentor.

Public names are defined in ``tools.ml_models.train.losses``.

Contains:
  - LossName: registered objective names.
  - LossSpec: pixel/Dice weights for one registered name.
  - dice_term / focal_term: the two imbalance-aware building blocks.
  - PlumeLoss: weighted BCE, Dice, and focal combination.
  - build_loss: construct a PlumeLoss from a name.
"""

from __future__ import annotations

from tools.ml_models.train.losses import (
    DEFAULT_FOCAL_ALPHA,
    DEFAULT_FOCAL_GAMMA,
    LOSS_NAMES,
    LOSS_SPECS,
    LossName,
    LossSpec,
    PlumeLoss,
    build_loss,
    dice_term,
    focal_term,
)

__all__ = [
    "DEFAULT_FOCAL_ALPHA",
    "DEFAULT_FOCAL_GAMMA",
    "LOSS_NAMES",
    "LOSS_SPECS",
    "LossName",
    "LossSpec",
    "PlumeLoss",
    "build_loss",
    "dice_term",
    "focal_term",
]
