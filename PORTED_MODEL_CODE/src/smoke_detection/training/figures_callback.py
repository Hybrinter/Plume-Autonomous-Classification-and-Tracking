"""End-of-training figure generation as a Lightning callback.

Accumulates epoch metrics from ``trainer.callback_metrics`` and, on
``on_train_end``, writes ``training_curves.png`` and ``val_predictions.png``
into the active logger's ``log_dir`` (typically
``lightning_logs/<experiment_name>/version_N/figures/``).

Replaces the Apr-16 argparse-era design (pre-Lightning) that mutated local
lists inside ``train_model()``. Under Lightning, the epoch loop is owned by
the Trainer, so accumulation hooks into ``on_*_epoch_end`` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from smoke_detection.common.logging import get_logger
from smoke_detection.training.classification_module import ClassificationModule
from smoke_detection.training.segmentation_module import SegmentationModule

log = get_logger(__name__)


def _rgb_composite(img_chw: np.ndarray) -> np.ndarray:
    """Build a contrast-stretched RGB composite from a 4-channel S2 tensor.

    Input channels are B2, B3, B4, B8 at indices 0..3. RGB uses B4, B3, B2
    (indices 2, 1, 0), matching the eval CLI visualization.
    """
    rgb = img_chw[[2, 1, 0], :, :].astype(np.float32)
    out = np.empty_like(rgb)
    for c in range(3):
        band = rgb[c]
        lo, hi = np.percentile(band, (2, 98))
        if hi <= lo:
            out[c] = np.zeros_like(band)
        else:
            out[c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return np.transpose(out, (1, 2, 0))


class TrainingFiguresCallback(Callback):
    """Accumulate per-epoch metrics and save summary figures at train end.

    Output directory is ``<logger.log_dir>/figures`` when a logger is
    attached; otherwise ``<trainer.default_root_dir>/figures``.
    """

    def __init__(self, num_val_samples: int = 9):
        super().__init__()
        self.num_val_samples = num_val_samples
        self.history: dict[str, list[float]] = {}

    def _record(self, metrics: dict[str, Any], keys: list[str]) -> None:
        for k in keys:
            if k in metrics:
                val = metrics[k]
                if isinstance(val, torch.Tensor):
                    val = float(val.detach().cpu().item())
                self.history.setdefault(k, []).append(float(val))

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        keys = ["train/loss_epoch", "train/acc", "train/iou"]
        self._record(dict(trainer.callback_metrics), keys)
        lr = _current_lr(trainer)
        if lr is not None:
            self.history.setdefault("lr", []).append(lr)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        keys = ["val/loss", "val/acc", "val/auc", "val/iou", "val/img_acc"]
        self._record(dict(trainer.callback_metrics), keys)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        out_dir = _resolve_out_dir(trainer)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._plot_training_curves(pl_module, out_dir / "training_curves.png")
        except Exception as exc:  # matplotlib/IO failures shouldn't break training
            log.warning("Failed to write training_curves.png: %s", exc)
        try:
            self._plot_val_predictions(trainer, pl_module, out_dir / "val_predictions.png")
        except Exception as exc:
            log.warning("Failed to write val_predictions.png: %s", exc)

    def _plot_training_curves(self, pl_module: L.LightningModule, out_path: Path) -> None:
        h = self.history
        if isinstance(pl_module, ClassificationModule):
            panels = [
                ("Loss", [("train/loss_epoch", "train"), ("val/loss", "val")], False),
                ("Accuracy", [("train/acc", "train"), ("val/acc", "val")], False),
                ("LR", [("lr", "lr")], True),
            ]
        elif isinstance(pl_module, SegmentationModule):
            panels = [
                ("Loss", [("train/loss_epoch", "train"), ("val/loss", "val")], False),
                ("IoU", [("train/iou", "train"), ("val/iou", "val")], False),
                ("Image acc (val)", [("val/img_acc", "val")], False),
                ("LR", [("lr", "lr")], True),
            ]
        else:
            return

        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
        if n == 1:
            axes = [axes]
        for ax, (title, series, log_y) in zip(axes, panels, strict=False):
            for key, label in series:
                ys = h.get(key, [])
                if not ys:
                    continue
                ax.plot(range(1, len(ys) + 1), ys, label=label, marker=".", linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("epoch")
            if log_y:
                ax.set_yscale("log")
            if len(series) > 1:
                ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info("Wrote %s", out_path)

    def _plot_val_predictions(
        self, trainer: L.Trainer, pl_module: L.LightningModule, out_path: Path
    ) -> None:
        samples = _collect_val_samples(trainer, self.num_val_samples)
        if not samples:
            log.info("Skipping val_predictions.png: no validation samples available")
            return
        pl_module.eval()
        with torch.no_grad():
            if isinstance(pl_module, ClassificationModule):
                self._plot_classification_grid(pl_module, samples, out_path)
            elif isinstance(pl_module, SegmentationModule):
                self._plot_segmentation_grid(pl_module, samples, out_path)
        log.info("Wrote %s", out_path)

    def _plot_classification_grid(
        self,
        pl_module: ClassificationModule,
        samples: list[dict[str, torch.Tensor]],
        out_path: Path,
    ) -> None:
        k = min(9, len(samples))
        rows = int(np.ceil(k / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
        axes = np.atleast_2d(axes)
        for i in range(rows * 3):
            ax = axes[i // 3, i % 3]
            ax.axis("off")
            if i >= k:
                continue
            s = samples[i]
            x = s["img"].unsqueeze(0).to(pl_module.device)
            logit = pl_module(x).squeeze()
            prob = float(torch.sigmoid(logit).item())
            pred = 1 if prob >= 0.5 else 0
            true = int(bool(s["lbl"]))
            ok = "OK" if pred == true else "X"
            rgb = _rgb_composite(s["img"].detach().cpu().numpy())
            ax.imshow(rgb)
            ax.set_title(
                f"True: {_label(true)} | Pred: {_label(pred)} [{ok}]  p={prob:.2f}", fontsize=9
            )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    def _plot_segmentation_grid(
        self,
        pl_module: SegmentationModule,
        samples: list[dict[str, torch.Tensor]],
        out_path: Path,
    ) -> None:
        k = min(9, len(samples))
        fig, axes = plt.subplots(k, 3, figsize=(9, 3 * k))
        axes = np.atleast_2d(axes)
        for i in range(k):
            s = samples[i]
            x = s["img"].unsqueeze(0).to(pl_module.device)
            y = s["fpt"].detach().cpu().numpy()
            logits = pl_module(x).squeeze().detach().cpu().numpy()
            pred = (logits >= 0).astype(np.float32)
            true_present = bool(y.sum() > 0)
            pred_present = bool(pred.sum() > 0)
            ok = "OK" if true_present == pred_present else "X"
            rgb = _rgb_composite(s["img"].detach().cpu().numpy())
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_ylabel(
                f"True: {_label(int(true_present))}\nPred: {_label(int(pred_present))} [{ok}]",
                fontsize=9,
            )
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
            axes[i, 1].imshow(y, cmap="Reds", vmin=0, vmax=1)
            axes[i, 1].set_title("Ground truth" if i == 0 else "")
            axes[i, 1].axis("off")
            axes[i, 2].imshow(pred, cmap="Greens", vmin=0, vmax=1)
            axes[i, 2].set_title("Prediction" if i == 0 else "")
            axes[i, 2].axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)


def _label(v: int) -> str:
    return "smoke" if v == 1 else "clear"


def _current_lr(trainer: L.Trainer) -> float | None:
    opts = trainer.optimizers
    if not opts:
        return None
    groups = opts[0].param_groups
    if not groups:
        return None
    return float(groups[0].get("lr", float("nan")))


def _resolve_out_dir(trainer: L.Trainer) -> Path:
    logger = trainer.logger
    base: Path
    if logger is not None and getattr(logger, "log_dir", None):
        base = Path(logger.log_dir)
    else:
        base = Path(trainer.default_root_dir or ".")
    return base / "figures"


def _collect_val_samples(trainer: L.Trainer, n: int) -> list[dict[str, torch.Tensor]]:
    dm = trainer.datamodule
    if dm is None:
        return []
    # Ensure val split is set up (safe no-op if already done).
    try:
        dm.setup(stage="fit")
    except Exception:
        pass
    try:
        loader = dm.val_dataloader()
    except Exception:
        return []
    out: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        bs = batch["img"].shape[0]
        for i in range(bs):
            item = {k: (v[i] if hasattr(v, "__getitem__") else v) for k, v in batch.items()}
            out.append(item)
            if len(out) >= n:
                return out
    return out
