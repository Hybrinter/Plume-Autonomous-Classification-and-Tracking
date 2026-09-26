"""Training objectives for the classifier and the segmentor.

Plume pixels are a small fraction of a scene, so plain BCE is dominated by easy
background and tends to settle on a conservative mask that scores poorly on the
IoU the acceptance gate actually measures. The alternatives here attack that
imbalance from two directions, and can be combined:

  - Dice optimises overlap directly, which is the same quantity IoU reports, so
    it is insensitive to how much background surrounds the plume.
  - Focal down-weights confidently-correct pixels, keeping the gradient on the
    ambiguous plume edge instead of the easy interior and sky.

Every loss consumes raw logits, matching the exported ONNX graphs, which do not
bake in a sigmoid. All of them work for both the segmentor's ``(N, 1, H, W)``
masks and the classifier's ``(N, 1)`` labels; on the latter, Dice reduces to a
soft F1.

Contains:
  - LossName: registered objective names.
  - LossSpec: pixel/Dice weights for one registered name.
  - dice_term / focal_term: the two imbalance-aware building blocks.
  - PlumeLoss: weighted BCE, Dice, and focal combination.
  - build_loss: construct a PlumeLoss from a name.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

LossName = Literal["bce", "dice", "bce_dice", "focal", "focal_dice"]

LOSS_NAMES: frozenset[str] = frozenset({"bce", "dice", "bce_dice", "focal", "focal_dice"})


@dataclass(frozen=True, slots=True)
class LossSpec:
    """Pixel and overlap weights for one registered objective.

    Attributes:
        use_focal: Use focal rather than BCE for the pixel term.
        dice_weight: Weight on the Dice term.
        pixel_weight: Weight on the pixel term.
    """

    use_focal: bool
    dice_weight: float
    pixel_weight: float


LOSS_SPECS: dict[str, LossSpec] = {
    "bce": LossSpec(use_focal=False, dice_weight=0.0, pixel_weight=1.0),
    "dice": LossSpec(use_focal=False, dice_weight=1.0, pixel_weight=0.0),
    "bce_dice": LossSpec(use_focal=False, dice_weight=1.0, pixel_weight=1.0),
    "focal": LossSpec(use_focal=True, dice_weight=0.0, pixel_weight=1.0),
    "focal_dice": LossSpec(use_focal=True, dice_weight=1.0, pixel_weight=1.0),
}

DEFAULT_FOCAL_GAMMA = 2.0
DEFAULT_FOCAL_ALPHA = 0.25
_DICE_SMOOTH = 1.0


def dice_term(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the soft Dice loss over each sample in a batch.

    Args:
        logits: torch.Tensor[float32, (N, ...)] raw logits.
        targets: torch.Tensor[float32, (N, ...)] targets in {0, 1}, same shape.

    Returns:
        torch.Tensor: Scalar ``1 - dice`` averaged over the batch. Smoothing by
        one in both numerator and denominator keeps an all-negative sample at
        zero loss when the prediction is also empty, rather than undefined.
    """
    probs = torch.sigmoid(logits).flatten(start_dim=1)
    flat = targets.flatten(start_dim=1).to(dtype=probs.dtype)
    intersection = (probs * flat).sum(dim=1)
    total = probs.sum(dim=1) + flat.sum(dim=1)
    dice = (2.0 * intersection + _DICE_SMOOTH) / (total + _DICE_SMOOTH)
    return (1.0 - dice).mean()


def focal_term(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = DEFAULT_FOCAL_GAMMA,
    alpha: float = DEFAULT_FOCAL_ALPHA,
) -> torch.Tensor:
    """Return the focal loss over a batch.

    Args:
        logits: torch.Tensor[float32] raw logits.
        targets: torch.Tensor[float32] targets in {0, 1}, same shape.
        gamma: Focusing exponent. Zero reduces this to weighted BCE.
        alpha: Positive-class weight in [0, 1]. Negatives take ``1 - alpha``.

    Returns:
        torch.Tensor: Scalar mean focal loss.
    """
    flat = targets.to(dtype=logits.dtype)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, flat, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * flat + (1.0 - probs) * (1.0 - flat)
    alpha_t = alpha * flat + (1.0 - alpha) * (1.0 - flat)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


class PlumeLoss(nn.Module):
    """Weighted sum of a pixel term and an overlap term.

    The pixel term is either BCE or focal; the overlap term is Dice. Weights of
    zero drop a term entirely, so one class covers every registered objective.
    """

    # Declared so the registered buffer reads back as a tensor rather than as
    # the Tensor-or-Module union nn.Module attribute access otherwise yields.
    pos_weight: torch.Tensor | None

    def __init__(
        self,
        use_focal: bool,
        dice_weight: float,
        pixel_weight: float,
        pos_weight: float = 0.0,
        focal_gamma: float = DEFAULT_FOCAL_GAMMA,
        focal_alpha: float = DEFAULT_FOCAL_ALPHA,
    ) -> None:
        """Build the objective.

        Args:
            use_focal: Use focal rather than BCE for the pixel term.
            dice_weight: Weight on the Dice term.
            pixel_weight: Weight on the pixel term.
            pos_weight: BCE positive-class weight. Ignored when ``use_focal``
                is set, which carries its own ``focal_alpha`` balance. Values
                at or below zero disable it.
            focal_gamma: Focal focusing exponent.
            focal_alpha: Focal positive-class weight.
        """
        super().__init__()
        self.use_focal = bool(use_focal)
        self.dice_weight = float(dice_weight)
        self.pixel_weight = float(pixel_weight)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        weight = (
            torch.tensor([float(pos_weight)], dtype=torch.float32) if pos_weight > 0.0 else None
        )
        self.register_buffer("pos_weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the scalar objective for one batch.

        Args:
            logits: torch.Tensor[float32] raw logits.
            targets: torch.Tensor[float32] targets in {0, 1}, same shape.

        Returns:
            torch.Tensor: Scalar loss.
        """
        total = torch.zeros((), dtype=logits.dtype, device=logits.device)
        if self.pixel_weight > 0.0:
            if self.use_focal:
                pixel = focal_term(logits, targets, self.focal_gamma, self.focal_alpha)
            else:
                weight = self.pos_weight
                pixel = nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    targets.to(dtype=logits.dtype),
                    pos_weight=weight.to(dtype=logits.dtype) if weight is not None else None,
                )
            total = total + self.pixel_weight * pixel
        if self.dice_weight > 0.0:
            total = total + self.dice_weight * dice_term(logits, targets)
        return total


def build_loss(
    name: str,
    pos_weight: float = 0.0,
    focal_gamma: float = DEFAULT_FOCAL_GAMMA,
    focal_alpha: float = DEFAULT_FOCAL_ALPHA,
) -> PlumeLoss:
    """Return the objective registered under ``name``.

    Args:
        name: One of ``bce``, ``dice``, ``bce_dice``, ``focal``, ``focal_dice``.
        pos_weight: BCE positive-class weight. Values at or below zero disable.
        focal_gamma: Focal focusing exponent.
        focal_alpha: Focal positive-class weight.

    Returns:
        PlumeLoss: Configured objective. Combined objectives weight both terms
        equally at 1.0, so ``bce_dice`` is a plain sum rather than an average.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    spec = LOSS_SPECS.get(name)
    if spec is None:
        raise ValueError(f"unknown loss {name!r}")
    return PlumeLoss(
        use_focal=spec.use_focal,
        dice_weight=spec.dice_weight,
        pixel_weight=spec.pixel_weight,
        pos_weight=pos_weight,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
