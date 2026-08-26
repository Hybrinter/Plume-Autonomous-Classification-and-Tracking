"""CLI entry point for training. Dispatches on ``config.task``."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from smoke_detection.common.logging import get_logger
from smoke_detection.common.seed import seed_everything
from smoke_detection.configs.base import BaseConfig
from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.loader import load_config
from smoke_detection.configs.segmentation import SegmentationConfig
from smoke_detection.data.classification_datamodule import ClassificationDataModule
from smoke_detection.data.segmentation_datamodule import SegmentationDataModule
from smoke_detection.training.classification_module import ClassificationModule
from smoke_detection.training.figures_callback import TrainingFiguresCallback
from smoke_detection.training.segmentation_module import SegmentationModule

log = get_logger(__name__)


def _build_classification(cfg: ClassificationConfig):
    module = ClassificationModule(
        in_channels=cfg.model.in_channels,
        pretrained=cfg.model.pretrained,
        lr=cfg.optim.lr,
        momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay,
        scheduler=cfg.optim.scheduler,
    )
    datamodule = ClassificationDataModule(
        data_root=cfg.paths.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        crop_size=cfg.data.crop_size,
        balance=cfg.data.balance,
    )
    monitor = "val/loss"
    return module, datamodule, monitor


def _build_segmentation(cfg: SegmentationConfig):
    module = SegmentationModule(
        in_channels=cfg.model.in_channels,
        n_classes=cfg.model.n_classes,
        bilinear=cfg.model.bilinear,
        lr=cfg.optim.lr,
        momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay,
        scheduler=cfg.optim.scheduler,
    )
    datamodule = SegmentationDataModule(
        data_root=cfg.paths.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        crop_size=cfg.data.crop_size,
    )
    monitor = "val/loss"
    return module, datamodule, monitor


def _build_trainer(cfg: BaseConfig, monitor: str) -> L.Trainer:
    logger = TensorBoardLogger(
        save_dir=str(cfg.paths.output_dir),
        name=cfg.paths.experiment_name,
    )
    checkpoint_cb = ModelCheckpoint(
        monitor=monitor,
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="epoch{epoch:03d}-{val/loss:.4f}",
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    figures_cb = TrainingFiguresCallback()
    return L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        deterministic=cfg.trainer.deterministic,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        fast_dev_run=cfg.trainer.fast_dev_run,
        default_root_dir=str(cfg.paths.output_dir),
        logger=logger,
        callbacks=[checkpoint_cb, lr_cb, figures_cb],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a smoke-detection model")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted override, e.g. --override optim.lr=1e-4 (repeatable)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.override)
    seed_everything(cfg.seed, deterministic=cfg.trainer.deterministic)
    log.info("Loaded config: task=%s experiment=%s", cfg.task, cfg.paths.experiment_name)

    if isinstance(cfg, ClassificationConfig):
        module, datamodule, monitor = _build_classification(cfg)
    elif isinstance(cfg, SegmentationConfig):
        module, datamodule, monitor = _build_segmentation(cfg)
    else:
        raise RuntimeError(f"Unsupported config type: {type(cfg).__name__}")

    trainer = _build_trainer(cfg, monitor=monitor)
    trainer.fit(module, datamodule=datamodule)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
