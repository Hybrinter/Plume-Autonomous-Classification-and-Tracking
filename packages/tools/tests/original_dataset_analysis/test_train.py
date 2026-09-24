"""The shared loop returns the best validation epoch."""

from __future__ import annotations

import torch
from tools.original_dataset_analysis.train import TrainConfig, run_training
from torch import nn
from torch.utils.data import DataLoader, Dataset


class _Tiles(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Four constant tiles."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        label = torch.tensor([1.0 if index % 2 == 0 else 0.0])
        image = torch.full((1, 8, 8), float(label))
        mask = image.clone()
        return image, label, mask


class _Head(nn.Module):
    """A 1x1 convolution used as both a classifier and a segmentor."""

    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plane = self.conv(x)
        if self.target == "label":
            return plane.mean(dim=(2, 3))
        return plane


def test_training_returns_a_checkpoint() -> None:
    """Two epochs on four tiles return a finite validation loss."""
    loader = DataLoader(_Tiles(), batch_size=2)
    model = _Head("label")
    result = run_training(
        model,
        loader,
        loader,
        TrainConfig(epochs=2, patience=2, lr=1e-2),
        target="label",
    )
    assert result.best_epoch in {0, 1}
    assert result.best_val_loss >= 0.0
    model.load_state_dict(result.state_dict)
