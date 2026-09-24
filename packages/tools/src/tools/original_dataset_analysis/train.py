"""Shared training loop for the classifier and the segmentor.

Contains:
  - TrainConfig: epochs, batch size, learning rate, and patience.
  - TrainResult: best validation loss and the epoch that reached it.
  - run_training: AdamW, cosine decay, and early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Loop settings shared by both tasks.

    Attributes:
        epochs: Maximum passes over the train loader.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        patience: Epochs without a new best validation loss before stopping.
        seed: Generator seed for dropout and shuffling.
    """

    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    seed: int = 0


@dataclass(frozen=True, slots=True)
class TrainResult:
    """The checkpoint with the lowest validation loss.

    Attributes:
        best_epoch: Zero-based epoch of the best validation loss.
        best_val_loss: That loss.
        state_dict: Weights at ``best_epoch``.
    """

    best_epoch: int
    best_val_loss: float
    state_dict: dict[str, torch.Tensor]


def _loss_of(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return binary cross-entropy with logits."""
    return nn.functional.binary_cross_entropy_with_logits(logits, target)


def _augment(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one shared flip or rotation to the image and the mask."""
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-2,))
        mask = torch.flip(mask, dims=(-2,))
    turns = int(torch.randint(0, 4, ()).item())
    if turns:
        image = torch.rot90(image, turns, dims=(-2, -1))
        mask = torch.rot90(mask, turns, dims=(-2, -1))
    return image, mask


def _mean_loss(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    target: str,
) -> float:
    """Return the mean loss of one loader."""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for image, label, mask in loader:
            image = image.to(device)
            batch_target = label.to(device) if target == "label" else mask.to(device)
            logits = model(image)
            if target == "label":
                logits = logits.reshape(batch_target.shape)
            loss = _loss_of(logits, batch_target)
            total += float(loss.item()) * image.shape[0]
            count += image.shape[0]
    return total / float(count) if count else 0.0


def run_training(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    config: TrainConfig,
    *,
    target: str,
    device: torch.device | None = None,
) -> TrainResult:
    """Train until patience expires or ``epochs`` is reached.

    Args:
        model: Classifier or segmentor. The classifier reads ``label``.
            The segmentor reads ``mask``.
        train_loader: Batches of image, label, and mask.
        val_loader: Held-out batches. Augmentation is not applied.
        config: Loop settings.
        target: ``label`` or ``mask``.
        device: Torch device. Defaults to CPU.

    Returns:
        TrainResult: Best validation loss and its weights.

    Raises:
        ValueError: If ``target`` is unknown or a loader is empty.
    """
    if target not in {"label", "mask"}:
        raise ValueError(f"target must be 'label' or 'mask'; got {target!r}")
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("train and val loaders must each yield a batch")
    torch.manual_seed(config.seed)
    chosen = device if device is not None else torch.device("cpu")
    model = model.to(chosen)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    best_loss = float("inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale = 0
    for epoch in range(config.epochs):
        model.train()
        for image, label, mask in train_loader:
            image = image.to(chosen)
            label = label.to(chosen)
            mask = mask.to(chosen)
            image, mask = _augment(image, mask)
            batch_target = label if target == "label" else mask
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            if target == "label":
                logits = logits.reshape(batch_target.shape)
            loss = _loss_of(logits, batch_target)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        schedule.step()
        val_loss = _mean_loss(model, val_loader, chosen, target)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    return TrainResult(best_epoch=best_epoch, best_val_loss=best_loss, state_dict=best_state)
