"""
pact.model.train — Training loop for the PACT plume segmentation model.

Satisfies: REQ-AIML-HIGH-001, REQ-AIML-HIGH-002

Implements one-epoch train and validation passes for the U-Net/ResNet-34 model.
Uses FocalLoss to address severe class imbalance (plume pixels are rare relative to
background). Optimizer: AdamW. LR schedule: CosineAnnealingLR.

All hyperparameters are encapsulated in TrainConfig — no magic numbers in loop bodies.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass, field

# third-party
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import FocalLoss


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameter configuration for model training.

    All fields have defaults that match the values intended for the HSG-AIML benchmark run.
    Override by constructing a new ``TrainConfig`` with the desired values — do not mutate.
    """

    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    val_split: float = 0.2
    seed: int = 42
    num_workers: int = 4
    save_best_path: str = "data/models/best.pt"


def train_epoch(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    optimizer: AdamW,
    criterion: FocalLoss,
    device: torch.device,
) -> float:
    """Run one full training epoch.

    Args:
        model:     The segmentation model (smp.Unet) in training mode.
        loader:    DataLoader yielding (tensor, mask) pairs.
        optimizer: AdamW optimiser — already constructed and attached to model params.
        criterion: FocalLoss instance (addresses class imbalance, REQ-AIML-HIGH-001).
        device:    Target torch device.

    Returns:
        Mean training loss over all batches in the epoch (float).
    """
    model.train()
    total_loss = 0.0
    for batch_tensor, batch_mask in loader:
        batch_tensor = batch_tensor.to(device)  # (B, 4, H, W) float32
        batch_mask = batch_mask.to(device)      # (B, 1, H, W) float32
        optimizer.zero_grad()
        logits = model(batch_tensor)            # (B, 1, H, W)
        loss = criterion(logits, batch_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: FocalLoss,
    device: torch.device,
) -> float:
    """Run one full validation epoch (no gradient computation).

    Args:
        model:     The segmentation model in eval mode (caller is responsible for setting).
        loader:    DataLoader for the validation split.
        criterion: Same FocalLoss instance used during training.
        device:    Target torch device.

    Returns:
        Mean validation loss over all batches in the epoch (float).
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_tensor, batch_mask in loader:
            batch_tensor = batch_tensor.to(device)  # (B, 4, H, W)
            batch_mask = batch_mask.to(device)      # (B, 1, H, W)
            logits = model(batch_tensor)            # (B, 1, H, W)
            loss = criterion(logits, batch_mask)
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)
