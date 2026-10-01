"""Training objectives for the classifier and the segmentor.

Public names are defined in ``tools.ml_models.train.losses``.

Contains:
  - LossName: registered objective names.
  - LossSpec: pixel/Dice weights for one registered name.
  - bce_per_sample / dice_per_sample / focal_per_sample: one value per image.
  - focal_dice_per_sample: per-image focal loss plus per-image Dice.
  - weighted_batch_loss: batch mean multiplied by a source weight.
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
    bce_per_sample,
    build_loss,
    dice_per_sample,
    dice_term,
    focal_dice_per_sample,
    focal_per_sample,
    focal_term,
    weighted_batch_loss,
)

__all__ = [
    "DEFAULT_FOCAL_ALPHA",
    "DEFAULT_FOCAL_GAMMA",
    "LOSS_NAMES",
    "LOSS_SPECS",
    "LossName",
    "LossSpec",
    "PlumeLoss",
    "bce_per_sample",
    "build_loss",
    "dice_per_sample",
    "dice_term",
    "focal_dice_per_sample",
    "focal_per_sample",
    "focal_term",
    "weighted_batch_loss",
]
