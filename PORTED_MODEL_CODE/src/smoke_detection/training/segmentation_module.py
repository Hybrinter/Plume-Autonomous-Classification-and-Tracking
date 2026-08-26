"""LightningModule for 4-channel smoke plume segmentation."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch import optim
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex

from smoke_detection.models.segmenter_unet import build_segmenter


class SegmentationModule(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 4,
        n_classes: int = 1,
        bilinear: bool = True,
        lr: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "none",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = build_segmenter(in_channels=in_channels, n_classes=n_classes, bilinear=bilinear)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_iou = BinaryJaccardIndex()
        self.val_iou = BinaryJaccardIndex()
        self.val_acc = BinaryAccuracy()
        self.test_iou = BinaryJaccardIndex()
        self.test_acc = BinaryAccuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _shared_step(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["img"]
        y = batch["fpt"].float().unsqueeze(1)  # [B, 1, H, W]
        logits = self.net(x)
        loss = self.loss_fn(logits, y)
        return loss, logits, y

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, logits, y = self._shared_step(batch)
        bs = batch["img"].shape[0]
        preds = (logits >= 0).int()
        self.train_iou.update(preds, y.int())
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log(
            "train/iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs
        )
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, logits, y = self._shared_step(batch)
        bs = batch["img"].shape[0]
        preds = (logits >= 0).int()
        self.val_iou.update(preds, y.int())
        image_pred = (preds.sum(dim=(1, 2, 3)) > 0).int()
        image_true = (y.sum(dim=(1, 2, 3)) > 0).int()
        self.val_acc.update(image_pred, image_true)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log(
            "val/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs
        )
        self.log("val/img_acc", self.val_acc, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        _, logits, y = self._shared_step(batch)
        bs = batch["img"].shape[0]
        preds = (logits >= 0).int()
        self.test_iou.update(preds, y.int())
        image_pred = (preds.sum(dim=(1, 2, 3)) > 0).int()
        image_true = (y.sum(dim=(1, 2, 3)) > 0).int()
        self.test_acc.update(image_pred, image_true)
        self.log("test/iou", self.test_iou, on_step=False, on_epoch=True, batch_size=bs)
        self.log("test/img_acc", self.test_acc, on_step=False, on_epoch=True, batch_size=bs)

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
