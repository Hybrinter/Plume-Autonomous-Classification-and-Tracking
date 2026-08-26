"""CLI entry point for evaluation. Runs ``trainer.test`` and dumps plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer

from smoke_detection.common.logging import get_logger
from smoke_detection.common.seed import seed_everything
from smoke_detection.configs.classification import ClassificationConfig
from smoke_detection.configs.loader import load_config
from smoke_detection.configs.segmentation import SegmentationConfig
from smoke_detection.data.classification_datamodule import ClassificationDataModule
from smoke_detection.data.segmentation_datamodule import SegmentationDataModule
from smoke_detection.evaluation.classification_metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
)
from smoke_detection.evaluation.segmentation_metrics import (
    plot_area_ratio_distribution,
    plot_iou_distribution,
)
from smoke_detection.training.classification_module import ClassificationModule
from smoke_detection.training.segmentation_module import SegmentationModule

log = get_logger(__name__)


def _eval_classification(cfg: ClassificationConfig, ckpt: Path, out_dir: Path) -> None:
    dm = ClassificationDataModule(
        data_root=cfg.paths.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        crop_size=cfg.data.crop_size,
        balance="none",
    )
    module = ClassificationModule.load_from_checkpoint(str(ckpt), weights_only=False)
    trainer = Trainer(accelerator=cfg.trainer.accelerator, devices=cfg.trainer.devices)
    trainer.test(module, datamodule=dm)

    dm.setup(stage="test")
    module.eval()
    scores: list[float] = []
    labels: list[int] = []
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for batch in dm.test_dataloader():
            logits = module(batch["img"].to(module.device)).cpu().squeeze(1)
            probs = torch.sigmoid(logits).tolist()
            ys = batch["lbl"].int().tolist()
            scores.extend(probs)
            labels.extend(ys)
            for p, y in zip(probs, ys, strict=False):
                pred = 1 if p >= 0.5 else 0
                if pred == 1 and y == 1:
                    tp += 1
                elif pred == 0 and y == 0:
                    tn += 1
                elif pred == 1 and y == 0:
                    fp += 1
                else:
                    fn += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(tp, tn, fp, fn, out_dir / "confusion_matrix.png")
    plot_roc_curve(scores, labels, out_dir / "roc_curve.png")
    log.info("Wrote classification eval plots to %s", out_dir)


def _eval_segmentation(cfg: SegmentationConfig, ckpt: Path, out_dir: Path) -> None:
    dm = SegmentationDataModule(
        data_root=cfg.paths.data_root,
        batch_size=1,
        num_workers=cfg.data.num_workers,
        crop_size=cfg.data.crop_size,
    )
    module = SegmentationModule.load_from_checkpoint(str(ckpt), weights_only=False)
    trainer = Trainer(accelerator=cfg.trainer.accelerator, devices=cfg.trainer.devices)
    trainer.test(module, datamodule=dm)

    dm.setup(stage="test")
    module.eval()
    ious: list[float] = []
    ratios: list[float] = []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            y = batch["fpt"].float().unsqueeze(1).to(module.device)
            logits = module(batch["img"].to(module.device))
            preds = (logits >= 0).float()
            inter = (preds * y).sum(dim=(1, 2, 3))
            union = ((preds + y) > 0).float().sum(dim=(1, 2, 3))
            for k in range(y.shape[0]):
                if y[k].sum() > 0 and preds[k].sum() > 0:
                    ious.append(float(inter[k] / union[k]))
                a_pred = float(preds[k].sum())
                a_true = float(y[k].sum())
                if a_pred == 0 and a_true == 0:
                    ratios.append(1.0)
                elif a_true == 0:
                    ratios.append(0.0)
                else:
                    ratios.append(a_pred / a_true)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_iou_distribution(ious, out_dir / "iou_distribution.png")
    plot_area_ratio_distribution(ratios, out_dir / "area_ratio_distribution.png")
    log.info(
        "Segmentation eval: mean IoU=%.4f mean area ratio=%.4f (n=%d)",
        float(np.mean(ious)) if ious else 0.0,
        float(np.mean(ratios)) if ratios else 0.0,
        len(ious),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained smoke-detection model")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument(
        "--override", action="append", default=[], help="Dotted overrides (repeatable)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for eval plots (defaults to <output_dir>/<experiment_name>/eval).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.override)
    seed_everything(cfg.seed, deterministic=cfg.trainer.deterministic)

    out_dir = args.out_dir or (cfg.paths.output_dir / cfg.paths.experiment_name / "eval")
    if isinstance(cfg, ClassificationConfig):
        _eval_classification(cfg, args.ckpt, out_dir)
    elif isinstance(cfg, SegmentationConfig):
        _eval_segmentation(cfg, args.ckpt, out_dir)
    else:
        raise RuntimeError(f"Unsupported config type: {type(cfg).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
