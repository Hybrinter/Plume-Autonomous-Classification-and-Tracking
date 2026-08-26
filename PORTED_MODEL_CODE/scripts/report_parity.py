"""Compare a trained checkpoint's test-set metrics against Mommert et al. 2020.

Reference numbers are taken from the paper abstract (4-channel model):

    Classification accuracy       94.3%
    Segmentation IoU              0.608
    Segmentation image accuracy   94.0%
    Mean |1 - area_ratio|         0.056   (paper says "within 5.6%")

Usage:

    python scripts/report_parity.py \\
        --config configs/classification/paper.yaml \\
        --ckpt lightning_logs/classification_4ch_resnet50_paper/version_0/checkpoints/last.ckpt

Prints a side-by-side table to stdout. Exits 0 regardless of parity; the
point of this script is reporting, not gating.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer

from smoke_detection.common.seed import seed_everything
from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.loader import load_config
from smoke_detection.configs.segmentation import SegmentationConfig
from smoke_detection.data.classification_datamodule import ClassificationDataModule
from smoke_detection.data.segmentation_datamodule import SegmentationDataModule
from smoke_detection.training.classification_module import ClassificationModule
from smoke_detection.training.segmentation_module import SegmentationModule

PAPER = {
    "classification_accuracy": 0.943,
    "classification_auc": None,  # not reported in abstract
    "segmentation_iou": 0.608,
    "segmentation_img_accuracy": 0.940,
    "segmentation_area_ratio_abs_error": 0.056,
}


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def _print_table(rows: list[tuple[str, float | None, float | None]]) -> None:
    w_name = max(len(r[0]) for r in rows)
    print(f"{'metric'.ljust(w_name)}   {'ours':>8}   {'paper':>8}   {'delta':>8}")
    print("-" * (w_name + 32))
    for name, ours, paper in rows:
        if ours is None or paper is None:
            delta_str = "—"
        else:
            delta_str = f"{ours - paper:+.4f}"
        print(f"{name.ljust(w_name)}   {_fmt(ours):>8}   {_fmt(paper):>8}   {delta_str:>8}")


def _parity_classification(cfg: ClassificationConfig, ckpt: Path) -> None:
    dm = ClassificationDataModule(
        data_root=cfg.paths.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        crop_size=cfg.data.crop_size,
        balance="none",
    )
    module = ClassificationModule.load_from_checkpoint(str(ckpt), weights_only=False)
    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=False,
        enable_checkpointing=False,
    )
    results = trainer.test(module, datamodule=dm, verbose=False)
    acc = results[0].get("test/acc") if results else None
    auc = results[0].get("test/auc") if results else None

    print("\n=== Classification parity (vs. Mommert et al. 2020, Table 1 / abstract) ===")
    _print_table(
        [
            ("test accuracy", float(acc) if acc is not None else None, PAPER["classification_accuracy"]),
            ("test AUC", float(auc) if auc is not None else None, PAPER["classification_auc"]),
        ]
    )


def _parity_segmentation(cfg: SegmentationConfig, ckpt: Path) -> None:
    dm = SegmentationDataModule(
        data_root=cfg.paths.data_root,
        batch_size=1,
        num_workers=cfg.data.num_workers,
        crop_size=cfg.data.crop_size,
    )
    module = SegmentationModule.load_from_checkpoint(str(ckpt), weights_only=False)
    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=False,
        enable_checkpointing=False,
    )
    results = trainer.test(module, datamodule=dm, verbose=False)
    iou = results[0].get("test/iou") if results else None
    img_acc = results[0].get("test/img_acc") if results else None

    # Area-ratio needs manual computation — mirror eval.py logic.
    dm.setup(stage="test")
    module.eval()
    abs_errors: list[float] = []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            y = batch["fpt"].float().unsqueeze(1).to(module.device)
            logits = module(batch["img"].to(module.device))
            preds = (logits >= 0).float()
            for k in range(y.shape[0]):
                a_pred = float(preds[k].sum())
                a_true = float(y[k].sum())
                if a_true == 0:
                    continue
                abs_errors.append(abs(1.0 - a_pred / a_true))
    mean_abs_err = float(np.mean(abs_errors)) if abs_errors else None

    print("\n=== Segmentation parity (vs. Mommert et al. 2020, Table 1 / abstract) ===")
    _print_table(
        [
            ("test IoU", float(iou) if iou is not None else None, PAPER["segmentation_iou"]),
            (
                "test image acc",
                float(img_acc) if img_acc is not None else None,
                PAPER["segmentation_img_accuracy"],
            ),
            (
                "mean |1 - area_ratio|",
                mean_abs_err,
                PAPER["segmentation_area_ratio_abs_error"],
            ),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report paper-parity metrics")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.override)
    seed_everything(cfg.seed, deterministic=cfg.trainer.deterministic)

    if isinstance(cfg, ClassificationConfig):
        _parity_classification(cfg, args.ckpt)
    elif isinstance(cfg, SegmentationConfig):
        _parity_segmentation(cfg, args.ckpt)
    else:
        raise RuntimeError(f"Unsupported config type: {type(cfg).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
