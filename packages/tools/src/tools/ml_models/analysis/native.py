"""Coarse segmentor logits scored on the native 120 px mask.

Contains:
  - NativeGridScores: Dice and positive-class IoU on the native grid.
  - score_on_native_grid: expand a coarse logit plane and score it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class NativeGridScores:
    """Overlap after a coarse map is expanded onto the 120 px mask.

    Attributes:
        dice: Positive-class Dice on the native grid.
        positive_iou: Positive-class IoU on the native grid.
    """

    dice: float
    positive_iou: float


def score_on_native_grid(logits: torch.Tensor, native_masks: torch.Tensor) -> NativeGridScores:
    """Expand a coarse logit plane onto the 120 px mask and score it.

    Args:
        logits: Shape ``(N, 1, H, W)``.
        native_masks: Shape ``(N, 120, 120)`` or ``(N, 1, 120, 120)``, values in {0, 1}.

    Returns:
        NativeGridScores: Dice and positive-class IoU on the native grid.
        The series name is ``native_dice``.

    Raises:
        ValueError: If the batch counts differ or the mask is not 120 px.
    """
    truth = native_masks.detach().float()
    if truth.ndim == 3:
        truth = truth[:, None, :, :]
    if truth.shape[0] != logits.shape[0] or truth.shape[-2:] != (120, 120):
        raise ValueError(
            f"native masks must be (N, 120, 120); got logits {tuple(logits.shape)} "
            f"and masks {tuple(truth.shape)}"
        )
    expanded = torch.nn.functional.interpolate(
        logits.detach().float(), size=(120, 120), mode="nearest"
    )
    predicted = torch.sigmoid(expanded) >= 0.5
    truth_bool = truth >= 0.5
    intersection = torch.logical_and(predicted, truth_bool).sum().item()
    union = torch.logical_or(predicted, truth_bool).sum().item()
    pred_sum = predicted.sum().item()
    truth_sum = truth_bool.sum().item()
    iou_pos = intersection / union if union else 1.0
    dice = 2.0 * intersection / (pred_sum + truth_sum) if pred_sum + truth_sum else 1.0
    return NativeGridScores(dice=float(dice), positive_iou=float(iou_pos))
