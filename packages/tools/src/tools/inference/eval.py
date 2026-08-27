"""Held-out split evaluation for a trained run directory.

Importing this module does not import torch. Torch loads inside `evaluate`
after the train extra is installed.

Contains:
  - evaluate: score a checkpoint on train, val, or test and write eval.json.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from tools.inference.metrics import classifier_metrics, compute_iou, segmentor_metrics, sigmoid
from tools.inference.train import TrainConfig, load_train_config

if TYPE_CHECKING:
    from torch import nn

    from tools.inference.data import ProcessedPack

_PREVIEW_LIMIT = 8


def _import_torch() -> ModuleType:
    """Import torch or raise a tools-extra error."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required for tools.inference.eval; install pact-tools[train]"
        ) from exc
    loaded: object = torch
    if not isinstance(loaded, ModuleType):
        raise TypeError("torch import did not return a module")
    return loaded


def _load_pack(run_dir: Path, cfg: TrainConfig) -> ProcessedPack:
    """Load the processed pack used by this run."""
    from tools.inference.data import load_processed_pack

    if cfg.data_dir:
        candidate = Path(cfg.data_dir)
        if (candidate / "splits.json").is_file():
            return load_processed_pack(candidate, bit_depth=cfg.bit_depth)
    synthetic = run_dir / "synthetic_pack"
    if (synthetic / "splits.json").is_file():
        return load_processed_pack(synthetic, bit_depth=cfg.bit_depth)
    raise FileNotFoundError(f"no processed pack with splits.json for run {run_dir}")


def _batch_indices(indices: tuple[int, ...], batch: int) -> list[tuple[int, ...]]:
    """Split indices into contiguous batches."""
    chunks: list[tuple[int, ...]] = []
    start = 0
    n = len(indices)
    while start < n:
        end = min(start + batch, n)
        chunks.append(indices[start:end])
        start = end
    return chunks


def _index_array(array: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """Return a contiguous float32 slice of ``array`` at ``indices``."""
    idx = np.asarray(indices, dtype=np.int64)
    return np.ascontiguousarray(array[idx], dtype=np.float32)


def _gather(
    model: nn.Module,
    torch: ModuleType,
    pack: ProcessedPack,
    indices: tuple[int, ...],
    kind: str,
    batch: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return logits, targets, and images for ``indices``."""
    model.eval()
    logit_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    image_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for chunk in _batch_indices(indices, batch):
            images = _index_array(pack.images, chunk)
            tensor = torch.from_numpy(images).to(device)
            logits = model(tensor)
            logit_chunks.append(np.asarray(logits.detach().cpu().numpy(), dtype=np.float32))
            image_chunks.append(images)
            if kind == "classifier":
                target_chunks.append(_index_array(pack.labels, chunk))
            else:
                target_chunks.append(_index_array(pack.masks, chunk))
    if not logit_chunks:
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 4, 1, 1), dtype=np.float32),
        )
    return (
        np.concatenate(logit_chunks, axis=0),
        np.concatenate(target_chunks, axis=0),
        np.concatenate(image_chunks, axis=0),
    )


def _metrics_dict(kind: str, logits: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
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


def _sample_scores(kind: str, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return per-sample quality in [0, 1] (accuracy or IoU)."""
    if logits.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if kind == "classifier":
        pred = logits.reshape(logits.shape[0], -1)[:, 0] >= 0.0
        label = targets.reshape(targets.shape[0], -1)[:, 0] >= 0.5
        return np.asarray(pred == label, dtype=np.float32)
    scores = np.zeros((logits.shape[0],), dtype=np.float32)
    probs = sigmoid(logits)
    for i in range(logits.shape[0]):
        pred_mask = probs[i, 0] if probs.ndim == 4 else probs[i]
        gold = targets[i, 0] if targets.ndim == 4 else targets[i]
        scores[i] = compute_iou(pred_mask, gold)
    return scores


def evaluate(
    run_dir: str | Path,
    checkpoint: str | None = None,
    split: str = "test",
    preview_limit: int = _PREVIEW_LIMIT,
) -> Path:
    """Score a checkpoint on a named split and write eval.json.

    Args:
        run_dir: Training run directory.
        checkpoint: Optional ``.pt`` path. None uses ``checkpoints/best.pt``.
        split: ``train``, ``val``, or ``test``.
        preview_limit: Max samples stored in ``predictions.npz`` for plots.

    Returns:
        Path: Path of the written ``eval.json``.

    Raises:
        ImportError: If torch is not installed.
        FileNotFoundError: If the run, checkpoint, or pack is missing.
        ValueError: If the split name is unknown.
    """
    root = Path(run_dir)
    cfg = load_train_config(str(root / "config.toml"))
    ckpt_path = Path(checkpoint) if checkpoint is not None else root / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    torch = _import_torch()
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")
    kind = str(payload.get("kind", cfg.kind))
    arch = str(payload.get("arch", cfg.arch))
    in_channels = int(payload.get("in_channels", cfg.in_channels))
    from tools.inference.arch.registry import build

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build(kind, arch, in_channels)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    pack = _load_pack(root, cfg)
    indices = pack.splits.for_name(split)
    batch = max(int(cfg.batch_size), 1)
    logits, targets, images = _gather(model, torch, pack, indices, kind, batch, device)
    metrics = _metrics_dict(kind, logits, targets)
    scores = _sample_scores(kind, logits, targets)
    preview = min(int(preview_limit), int(images.shape[0]))
    order = np.argsort(scores)[:preview] if preview else np.zeros((0,), dtype=np.int64)
    np.savez(
        root / "predictions.npz",
        images=images[order] if preview else images,
        targets=targets[order] if preview else targets,
        logits=logits[order] if preview else logits,
        scores=scores[order] if preview else scores,
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
