"""Late fusion of a separately trained VIS model and SWIR model.

Contains:
  - DEFAULT_VIS_WEIGHTS: VIS weights searched on validation (0 to 1 in 0.1 steps).
  - LateFusion: weighted sum of the two models' logits on one paired input.
  - select_weight: the VIS weight with the best validation score.

The paired input stacks BLUE, GREEN, RED, SWIR in that plane order. The VIS
model reads the first ``vis_planes`` planes and the SWIR model reads the rest.
Both models take the same GSD encoding. Weight 1 is VIS only and weight 0 is
SWIR only, so the full grid includes both single-camera ends.

Satisfies: REQ-AIML-COMP-001.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

DEFAULT_VIS_WEIGHTS: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(11))


class LateFusion(nn.Module):
    """Weighted logit sum ``w * vis(x[:, :k]) + (1 - w) * swir(x[:, k:])``.

    Args:
        vis: VIS model called as ``vis(image, gsd)``.
        swir: SWIR model called as ``swir(image, gsd)``.
        vis_weight: VIS weight ``w`` in ``[0, 1]``.
        vis_planes: VIS plane count ``k`` at the front of the input.

    Raises:
        ValueError: If ``vis_weight`` is outside ``[0, 1]`` or ``vis_planes < 1``.
    """

    def __init__(
        self, vis: nn.Module, swir: nn.Module, vis_weight: float = 0.5, vis_planes: int = 3
    ) -> None:
        super().__init__()
        if vis_planes < 1:
            raise ValueError("vis_planes must be at least 1")
        self.vis = vis
        self.swir = swir
        self.vis_planes = vis_planes
        self.vis_weight = 0.5
        self.set_vis_weight(vis_weight)

    def set_vis_weight(self, vis_weight: float) -> None:
        """Set the VIS weight.

        Args:
            vis_weight: VIS weight in ``[0, 1]``.

        Raises:
            ValueError: If the weight is outside ``[0, 1]``.
        """
        if not 0.0 <= vis_weight <= 1.0:
            raise ValueError(f"vis_weight must be in [0, 1], got {vis_weight}")
        self.vis_weight = vis_weight

    def forward(self, image: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        """Return the fused logits.

        Args:
            image: Paired input ``(N, C, H, W)``, VIS planes first.
            gsd: GSD encoding passed unchanged to both models.

        Returns:
            torch.Tensor: Fused logits with the shape of each model's output.
        """
        k = self.vis_planes
        vis_out: torch.Tensor = self.vis(image[:, :k], gsd)
        swir_out: torch.Tensor = self.swir(image[:, k:], gsd)
        return self.vis_weight * vis_out + (1.0 - self.vis_weight) * swir_out


def select_weight(val_scores: Mapping[float, float]) -> float:
    """Return the VIS weight with the highest validation score.

    Args:
        val_scores: Validation score per VIS weight; higher is better.

    Returns:
        float: The best weight. Ties keep the larger VIS weight.

    Raises:
        ValueError: If ``val_scores`` is empty.
    """
    if not val_scores:
        raise ValueError("select_weight needs at least one score")
    return max(val_scores, key=lambda w: (val_scores[w], w))
