"""Tests for late fusion of VIS and SWIR models."""

from __future__ import annotations

import pytest
import torch
from tools.ml_models.fusion.late import DEFAULT_VIS_WEIGHTS, LateFusion, select_weight
from torch import nn


class _PlaneSum(nn.Module):
    """Return the per-sample sum of the input planes it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.planes = 0

    def forward(self, image: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        """Record the plane count and sum the input."""
        self.planes = image.shape[1]
        return image.sum(dim=(1, 2, 3)).unsqueeze(1) + 0.0 * gsd.sum()


def test_late_fusion_splits_planes_and_weights_logits() -> None:
    """VIS reads 3 planes, SWIR reads the rest, and logits mix by the weight."""
    vis, swir = _PlaneSum(), _PlaneSum()
    image = torch.zeros(2, 4, 2, 2)
    image[:, :3] = 1.0
    image[:, 3] = 2.0
    model = LateFusion(vis, swir, vis_weight=0.25)
    out = model(image, torch.zeros(2, 2))
    assert vis.planes == 3
    assert swir.planes == 1
    assert torch.allclose(out, torch.full((2, 1), 0.25 * 12.0 + 0.75 * 8.0))


def test_weight_bounds() -> None:
    """Weights outside [0, 1] and an empty VIS slice are rejected."""
    with pytest.raises(ValueError):
        LateFusion(_PlaneSum(), _PlaneSum(), vis_weight=1.5)
    with pytest.raises(ValueError):
        LateFusion(_PlaneSum(), _PlaneSum(), vis_planes=0)
    model = LateFusion(_PlaneSum(), _PlaneSum())
    with pytest.raises(ValueError):
        model.set_vis_weight(-0.1)


def test_default_grid_covers_both_single_camera_ends() -> None:
    """The default grid runs 0 to 1 in 0.1 steps."""
    assert DEFAULT_VIS_WEIGHTS[0] == 0.0
    assert DEFAULT_VIS_WEIGHTS[-1] == 1.0
    assert len(DEFAULT_VIS_WEIGHTS) == 11


def test_select_weight_prefers_score_then_larger_weight() -> None:
    """The best score wins; a tie keeps the larger VIS weight."""
    assert select_weight({0.3: 0.7, 0.6: 0.8, 0.9: 0.75}) == 0.6
    assert select_weight({0.4: 0.8, 0.7: 0.8}) == 0.7
    with pytest.raises(ValueError):
        select_weight({})
