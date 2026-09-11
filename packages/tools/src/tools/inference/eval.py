"""Held-out split evaluation for a trained run directory.

Contains:
  - evaluate: score a checkpoint on train, val, or test and write eval.json.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tools.inference.arch.registry import build
from tools.inference.data import ProcessedPack, SplitDataset, load_processed_pack
from tools.inference.metrics import classifier_metrics, compute_iou, segmentor_metrics, sigmoid
from tools.inference.train import TrainConfig, load_train_config

_PREVIEW_LIMIT = 8


def _load_pack(run_dir: Path, cfg: TrainConfig) -> ProcessedPack:
    """Load the processed pack used by this run."""
    if cfg.data_dir:
        candidate = Path(cfg.data_dir)
        if (candidate / "splits.json").is_file():
            return load_processed_pack(
                candidate, bit_depth=cfg.bit_depth, load_masks=cfg.kind != "classifier"
            )
    synthetic = run_dir / "synthetic_pack"
    if (synthetic / "splits.json").is_file():
        return load_processed_pack(
            synthetic, bit_depth=cfg.bit_depth, load_masks=cfg.kind != "classifier"
        )
    raise FileNotFoundError(f"no processed pack with splits.json for run {run_dir}")


def _gather(
    model: nn.Module,
    pack: ProcessedPack,
    kind: str,
    split: str,
    batch: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logits, targets, and images for one named split."""
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        SplitDataset(pack, kind, split),
        batch_size=batch,
        shuffle=False,
    )
    model.eval()
    logit_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    image_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            logit_chunks.append(logits.detach().cpu())
            target_chunks.append(targets.cpu())
            image_chunks.append(images.cpu())
    if not logit_chunks:
        empty = torch.zeros((0, 1), dtype=torch.float32)
        return empty, empty, torch.zeros((0, 4, 1, 1), dtype=torch.float32)
    return (
        torch.cat(logit_chunks, dim=0),
        torch.cat(target_chunks, dim=0),
        torch.cat(image_chunks, dim=0),
    )


def _metrics_dict(kind: str, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    """Return JSON-ready metric fields for one split."""
    if kind == "classifier":
        report = classifier_metrics(logits, targets)
        return {
            "n": report.n,
            "accuracy": report.accuracy,
            "precision": report.precision,
            "recall": report.recall,
            "f1": report.f1,
            "roc_auc": report.roc_auc,
            "pr_auc": report.pr_auc,
            "brier": report.brier,
            "bce": report.bce,
        }
    report_seg = segmentor_metrics(logits, targets)
    return {
        "n": report_seg.n,
        "mean_iou": report_seg.mean_iou,
        "mean_dice": report_seg.mean_dice,
        "mean_iou_blob_gate": report_seg.mean_iou_blob_gate,
        "bce": report_seg.bce,
    }


def _sample_scores(kind: str, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return per-sample quality in [0, 1] (accuracy or IoU)."""
    if logits.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.float32)
    if kind == "classifier":
        pred = logits.reshape(logits.shape[0], -1)[:, 0] >= 0.0
        label = targets.reshape(targets.shape[0], -1)[:, 0] >= 0.5
        return (pred == label).to(dtype=torch.float32)
    scores = torch.zeros((logits.shape[0],), dtype=torch.float32)
    probs = sigmoid(logits)
    for i in range(logits.shape[0]):
        pred_mask = probs[i, 0] if probs.ndim == 4 else probs[i]
        gold = targets[i, 0] if targets.ndim == 4 else targets[i]
        scores[i] = compute_iou(pred_mask, gold)
    return scores


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Copy a CPU tensor to a contiguous float32 ndarray for npz/plots."""
    return np.ascontiguousarray(tensor.detach().cpu().numpy())


def evaluate(
    run_dir: str | Path,
    checkpoint: str | None = None,
    split: str = "val",
    preview_limit: int = _PREVIEW_LIMIT,
) -> Path:
    """Score a checkpoint on a named split and write eval.json.

    Args:
        run_dir: Training run directory.
        checkpoint: Optional ``.pt`` path. None uses ``checkpoints/best.pt``.
        split: ``train``, ``val``, or ``test``. Default ``val``.
        preview_limit: Max samples stored in ``predictions.npz`` for plots.

    Returns:
        Path: Path of the written ``eval.json``.

    Raises:
        FileNotFoundError: If the run, checkpoint, or pack is missing.
        ValueError: If the split name is unknown.
    """
    root = Path(run_dir)
    cfg = load_train_config(str(root / "config.toml"))
    ckpt_path = Path(checkpoint) if checkpoint is not None else root / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")
    kind = str(payload.get("kind", cfg.kind))
    arch = str(payload.get("arch", cfg.arch))
    in_channels = int(payload.get("in_channels", cfg.in_channels))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build(kind, arch, in_channels)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    pack = _load_pack(root, cfg)
    indices = pack.splits.for_name(split)
    batch = max(int(cfg.batch_size), 1)
    if not indices:
        logits = torch.zeros((0, 1), dtype=torch.float32)
        targets = torch.zeros((0, 1), dtype=torch.float32)
        images = torch.zeros(
            (0, int(pack.images.shape[1]), int(pack.images.shape[2]), int(pack.images.shape[3])),
            dtype=torch.float32,
        )
    else:
        logits, targets, images = _gather(model, pack, kind, split, batch, device)
    metrics = _metrics_dict(kind, logits, targets)
    scores = _sample_scores(kind, logits, targets)
    preview = min(int(preview_limit), int(images.shape[0]))
    order = torch.argsort(scores)[:preview] if preview else torch.zeros((0,), dtype=torch.int64)
    np.savez(
        root / "predictions.npz",
        images=_to_numpy(images[order]) if preview else _to_numpy(images),
        targets=_to_numpy(targets[order]) if preview else _to_numpy(targets),
        logits=_to_numpy(logits[order]) if preview else _to_numpy(logits),
        scores=_to_numpy(scores[order]) if preview else _to_numpy(scores),
        kind=np.array(kind),
    )
    report = {
        "split": split,
        "checkpoint": str(ckpt_path),
        "kind": kind,
        "arch": arch,
        **metrics,
    }
    eval_path = root / "eval.json"
    eval_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary_path = root / "summary.json"
    summary: dict[str, object] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key, value in metrics.items():
        summary[f"test_{key}" if split == "test" else f"{split}_{key}"] = value
    summary["eval_split"] = split
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return eval_path
