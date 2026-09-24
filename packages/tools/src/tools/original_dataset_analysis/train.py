"""Shared training loop for the classifier and the segmentor.

Contains:
  - TrainConfig: epochs, batch size, learning rate, and patience.
  - TrainResult: best validation loss and the epoch that reached it.
  - run_training: AdamW, cosine decay, and early stopping.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

_Batch = tuple[torch.Tensor, ...]


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


def _parts(
    batch: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Split a collated batch into image, label, mask, and an optional flag.

    Args:
        batch: Three tensors, or four when the dataset reports annotation flags.

    Returns:
        tuple: Image, label, mask, and the annotation flag or ``None``.

    Raises:
        ValueError: If ``batch`` does not hold three or four tensors.
    """
    if len(batch) not in {3, 4}:
        raise ValueError(f"batch must hold 3 or 4 tensors; got {len(batch)}")
    annotated = batch[3] if len(batch) == 4 else None
    return batch[0], batch[1], batch[2], annotated


def _drop_unannotated(
    image: torch.Tensor,
    label: torch.Tensor,
    mask: torch.Tensor,
    annotated: torch.Tensor | None,
    *,
    target: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Drop tiles that have no annotation file when the target is the mask.

    Args:
        image: Batch images.
        label: Presence labels.
        mask: Raster masks.
        annotated: Flag with shape ``(N,)`` or ``(N, 1)``. ``None`` keeps the batch.
        target: ``label`` or ``mask``.

    Returns:
        The kept tensors, or ``None`` when no tile remains.
    """
    if target != "mask" or annotated is None:
        return image, label, mask
    keep = annotated.reshape(-1) > 0
    if int(keep.sum().item()) == 0:
        return None
    return image[keep], label[keep], mask[keep]


def _mean_loss(
    model: nn.Module,
    loader: DataLoader[_Batch],
    device: torch.device,
    target: str,
) -> float:
    """Return the mean loss of one loader."""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            image, label, mask, annotated = _parts(batch)
            kept = _drop_unannotated(image, label, mask, annotated, target=target)
            if kept is None:
                continue
            image, label, mask = kept
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
    train_loader: DataLoader[_Batch],
    val_loader: DataLoader[_Batch],
    config: TrainConfig,
    *,
    target: str,
    device: torch.device | None = None,
) -> TrainResult:
    """Train until patience expires or ``epochs`` is reached.

    Args:
        model: Classifier or segmentor. The classifier reads ``label``.
            The segmentor reads ``mask``.
        train_loader: Batches of image, label, and mask. A fourth tensor is an
            annotation flag.
        val_loader: Held-out batches. Augmentation is not applied. A fourth
            tensor is an annotation flag.
        config: Loop settings.
        target: ``label`` or ``mask``.
        device: Torch device. Defaults to CPU.

    Returns:
        TrainResult: Best validation loss and its weights.

    Raises:
        ValueError: If ``target`` is unknown, a loader is empty, or a batch
            does not hold 3 or 4 tensors.
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
        stepped = False
        for batch in train_loader:
            image, label, mask, annotated = _parts(batch)
            kept = _drop_unannotated(image, label, mask, annotated, target=target)
            if kept is None:
                continue
            image, label, mask = kept
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
            stepped = True
        if stepped:
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
