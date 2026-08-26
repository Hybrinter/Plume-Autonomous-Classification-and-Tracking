"""LightningDataModule wrapping ``SmokePlumeDataset`` for classification."""

from __future__ import annotations

import platform
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, RandomSampler

from smoke_detection.common.paths import classification_split
from smoke_detection.data.classification_dataset import (
    SmokePlumeDataset,
    build_default_transform,
    build_eval_transform,
)


class ClassificationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        crop_size: int = 90,
        balance: str = "upsample",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_root = Path(data_root).resolve()
        self.batch_size = batch_size
        self.num_workers = 0 if platform.system() == "Windows" else num_workers
        self.crop_size = crop_size
        self.balance = balance

    def setup(self, stage: str | None = None) -> None:
        train_tfm = build_default_transform(crop_size=self.crop_size)
        eval_tfm = build_eval_transform()
        if stage in (None, "fit"):
            self.train_ds = SmokePlumeDataset(
                datadir=classification_split("train", self.data_root),
                transform=train_tfm,
                balance=self.balance,
            )
            self.val_ds = SmokePlumeDataset(
                datadir=classification_split("val", self.data_root),
                transform=eval_tfm,
                balance="none",
            )
        if stage in (None, "test", "predict"):
            self.test_ds = SmokePlumeDataset(
                datadir=classification_split("test", self.data_root),
                transform=eval_tfm,
                balance="none",
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
