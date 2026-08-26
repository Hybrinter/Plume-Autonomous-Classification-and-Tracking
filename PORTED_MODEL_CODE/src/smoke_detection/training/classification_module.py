"""LightningModule for 4-channel smoke plume classification."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch import optim
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from smoke_detection.models.classifier_resnet import build_classifier


class ClassificationModule(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 4,
        pretrained: bool = True,
        lr: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "none",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = build_classifier(in_channels=in_channels, pretrained=pretrained)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_auc = BinaryAUROC()
        self.test_acc = BinaryAccuracy()
        self.test_auc = BinaryAUROC()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _shared_step(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["img"]
        y = batch["lbl"].float().reshape(-1, 1)
        logits = self.net(x)
        loss = self.loss_fn(logits, y)
        return loss, logits, y

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, logits, y = self._shared_step(batch)
        bs = batch["img"].shape[0]
        self.train_acc.update(torch.sigmoid(logits).squeeze(1), y.squeeze(1).int())
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs
        )
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, logits, y = self._shared_step(batch)
        bs = batch["img"].shape[0]
        probs = torch.sigmoid(logits).squeeze(1)
        self.val_acc.update(probs, y.squeeze(1).int())
        self.val_auc.update(probs, y.squeeze(1).int())
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log(
            "val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs
        )
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        _, logits, y = self._shared_step(batch)
        bs = batch["img"].shape[0]
        probs = torch.sigmoid(logits).squeeze(1)
        self.test_acc.update(probs, y.squeeze(1).int())
        self.test_auc.update(probs, y.squeeze(1).int())
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, batch_size=bs)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, batch_size=bs)

    def configure_optimizers(self):
        opt = optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.scheduler == "plateau":
            sched = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, threshold=1e-4, min_lr=1e-6
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val/loss"}}
        if self.hparams.scheduler == "cosine":
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt
