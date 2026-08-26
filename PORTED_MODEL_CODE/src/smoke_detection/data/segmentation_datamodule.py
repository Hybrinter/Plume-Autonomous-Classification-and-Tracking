"""LightningDataModule wrapping ``SmokePlumeSegmentationDataset``."""

from __future__ import annotations

import platform
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, RandomSampler

from smoke_detection.common.paths import segmentation_split
from smoke_detection.data.segmentation_dataset import (
    SmokePlumeSegmentationDataset,
    build_default_transform,
    build_eval_transform,
)


class SegmentationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: Path,
        batch_size: int = 16,
        num_workers: int = 4,
        crop_size: int = 90,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_root = Path(data_root).resolve()
        self.batch_size = batch_size
        self.num_workers = 0 if platform.system() == "Windows" else num_workers
        self.crop_size = crop_size

    def setup(self, stage: str | None = None) -> None:
        train_tfm = build_default_transform(crop_size=self.crop_size)
        eval_tfm = build_eval_transform()
        if stage in (None, "fit"):
            tr_img, tr_lbl = segmentation_split("train", self.data_root)
            va_img, va_lbl = segmentation_split("val", self.data_root)
            self.train_ds = SmokePlumeSegmentationDataset(
                datadir=tr_img, seglabeldir=tr_lbl, transform=train_tfm
            )
            self.val_ds = SmokePlumeSegmentationDataset(
                datadir=va_img, seglabeldir=va_lbl, transform=eval_tfm
            )
        if stage in (None, "test", "predict"):
            te_img, te_lbl = segmentation_split("test", self.data_root)
            self.test_ds = SmokePlumeSegmentationDataset(
                datadir=te_img, seglabeldir=te_lbl, transform=eval_tfm
            )

    def train_dataloader(self) -> DataLoader:
        sampler = RandomSampler(
            self.train_ds, replacement=True, num_samples=max(1, 2 * len(self.train_ds) // 3)
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
