"""The shared loop returns the best validation epoch."""

from __future__ import annotations

import pytest
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
        plane: torch.Tensor = self.conv(x)
        if self.target == "label":
            reduced: torch.Tensor = plane.mean(dim=(2, 3))
            return reduced
        return plane


class _Unannotated(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Tiles whose annotation flag is 0 and whose mask is all ones."""

    def __len__(self) -> int:
        return 4

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.zeros(1, 8, 8)
        label = torch.tensor([0.0])
        mask = torch.ones(1, 8, 8)
        annotated = torch.tensor([0.0])
        return image, label, mask, annotated


def test_segmentor_skips_tiles_without_annotations() -> None:
    """A mask target leaves weights unchanged when every train flag is 0."""
    model = _Head("mask")
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    result = run_training(
        model,
        DataLoader(_Unannotated(), batch_size=2),
        DataLoader(_Tiles(), batch_size=2),
        TrainConfig(epochs=2, patience=2, lr=1e-2),
        target="mask",
    )
    for key, value in result.state_dict.items():
        assert torch.equal(value, before[key])


def test_segmentor_rejects_validation_without_annotations() -> None:
    """A mask target rejects a validation loader whose flags are all 0."""
    with pytest.raises(ValueError, match="usable"):
        run_training(
            _Head("mask"),
            DataLoader(_Tiles(), batch_size=2),
            DataLoader(_Unannotated(), batch_size=2),
            TrainConfig(epochs=2, patience=2, lr=1e-2),
            target="mask",
        )


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
    assert len(result.history) == 2
    assert result.history[0].epoch == 1
    assert result.history[0].test_loss is None
    assert result.history[0].train_loss is not None
    model.load_state_dict(result.state_dict)
