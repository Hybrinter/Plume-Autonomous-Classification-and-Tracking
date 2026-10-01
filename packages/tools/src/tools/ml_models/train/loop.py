"""Training loop for GSD-conditioned models over finished datasets.

A run owns one directory under ``TrainConfig.run_dir`` named by ``run_id`` or a
kind/digest/timestamp default. The directory is created once and an existing
one is rejected; ``overwrite`` is a reserved flag and never makes the loop
destructive.

The loader, not the loop, owns sampling, augmentation, and normalisation.
Dataset weights influence only which dataset a batch is drawn from; they do
not scale the loss. Validation uses :func:`evaluate`, which scores every row
of every selected shard once, so the best checkpoint is selected strictly on
the combined validation metric.

Contains:
  - EXPORT_HEIGHT_PX / EXPORT_WIDTH_PX: export trace defaults carried in
    checkpoint metadata; they are not training resize dimensions.
  - train: the public ``Result`` boundary.
  - _train: the raising implementation.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from flight.libs.types import Err, Ok, Result
from torch import nn

from tools.ml_models.arch.film import CONDITIONING_ID, IGNORED_CONDITIONING_ID, IgnoreGsd
from tools.ml_models.arch.registry import build as build_model
from tools.ml_models.arch.registry import resolve_arch
from tools.ml_models.dataset.build import build_synthetic
from tools.ml_models.dataset.loader import make_loader
from tools.ml_models.dataset.manifest import DatasetManifest, check_compatible, load_manifest
from tools.ml_models.train.config import TrainConfig, config_digest, write_train_config_toml
from tools.ml_models.train.evaluate import evaluate
from tools.ml_models.train.losses import build_loss
from tools.ml_models.train.provenance import training_provenance

EXPORT_HEIGHT_PX = 193
EXPORT_WIDTH_PX = 258

# Validation metrics that improve by shrinking; every other metric maximises.
_MINIMIZE_METRICS = frozenset({"bce", "brier"})


def train(cfg: TrainConfig | None = None) -> Result[Path, str]:
    """Train one model on finished datasets and return the run directory.

    Args:
        cfg: Training configuration. None uses :class:`TrainConfig` defaults,
            which build a small synthetic dataset inside the run directory.

    Returns:
        Result[Path, str]: Ok with the run directory on success.

    """
    resolved = cfg if cfg is not None else TrainConfig()
    try:
        return Ok(_train(resolved))
    except (OSError, ValueError, RuntimeError) as exc:
        return Err(str(exc))


def _train(cfg: TrainConfig) -> Path:
    """Run one training pass; raises on any failure."""
    run_id = cfg.run_id or f"{cfg.kind}-{config_digest(cfg)}-{time.time_ns()}"
    run = Path(cfg.run_dir) / run_id
    if run.exists():
        raise FileExistsError(f"run directory {run} already exists")
    run.mkdir(parents=True)
    (run / "checkpoints").mkdir()
    write_train_config_toml(run / "config.toml", cfg)

    if cfg.datasets:
        dests: list[str | Path] = list(cfg.datasets)
    else:
        dests = [run / "synthetic"]
        build_synthetic(dests[0], n=cfg.synthetic_samples, seed=cfg.seed)

    manifests = [load_manifest(Path(dest) / "dataset.json") for dest in dests]
    check_compatible(manifests)
    provenance = training_provenance(dests, manifests, cfg.kind)
    weights: tuple[float, ...] = cfg.dataset_weights if cfg.dataset_weights else (1.0,) * len(dests)
    _require_train_val(dests, manifests, cfg.kind)

    torch.manual_seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.kind, cfg.arch, cast(int, provenance["in_channels"]))
    model.to(device)

    if cfg.optimizer == "sgd":
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        if cfg.scheduler == "cosine"
        else None
    )
    criterion = build_loss(
        cfg.loss,
        pos_weight=cfg.pos_weight,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=cfg.focal_alpha,
    )
    scaler = torch.amp.GradScaler("cuda") if cfg.amp and device.startswith("cuda") else None

    metric = cfg.val_metric or ("f1" if cfg.kind == "classifier" else "mean_iou")
    minimize = metric in _MINIMIZE_METRICS
    arch = resolve_arch(cfg.kind, cfg.arch)
    conditioning = IGNORED_CONDITIONING_ID if isinstance(model, IgnoreGsd) else CONDITIONING_ID
    dataset_hash = hashlib.sha256(
        "|".join(manifest.dataset_hash for manifest in manifests).encode()
    ).hexdigest()

    history: list[dict[str, object]] = []
    best = math.inf if minimize else -math.inf
    evaluations_without_improvement = 0
    step = 0
    epoch = 0
    stop = False
    last_report: dict[str, object] | None = None
    while epoch < cfg.epochs and not stop:
        epoch += 1
        model.train()
        n_batches = math.ceil(cast(int, provenance["train_samples"]) / cfg.batch_size)
        loader = make_loader(
            dests,
            cfg.kind,
            "train",
            cfg.batch_size,
            weights,
            cfg.seed + epoch,
            n_batches=n_batches,
        )
        for images, gsd, targets in loader:
            optimizer.zero_grad()
            if scaler is not None:
                with torch.autocast("cuda"):
                    output: torch.Tensor = model(images.to(device), gsd.to(device))
                    if output.shape != targets.shape:
                        raise ValueError(
                            f"model output {tuple(output.shape)} != target {tuple(targets.shape)}"
                        )
                    loss = criterion(output, targets.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(images.to(device), gsd.to(device))
                if output.shape != targets.shape:
                    raise ValueError(
                        f"model output {tuple(output.shape)} != target {tuple(targets.shape)}"
                    )
                loss = criterion(output, targets.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}")
                loss.backward()
                optimizer.step()
            step += 1
            if cfg.max_steps is not None and step >= cfg.max_steps:
                stop = True
                break
        if scheduler is not None:
            scheduler.step()

        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs or stop:
            last_report = evaluate(
                model, dests, manifests, cfg.kind, "val", cfg.batch_size, weights, device
            )
            combined = cast(dict[str, float], last_report["combined"])
            value = float(combined[metric])
            history.append({"epoch": epoch, "step": step, "validation": combined})
            improved = value < best if minimize else value > best
            if improved:
                best = value
                evaluations_without_improvement = 0
                _save_checkpoint(
                    run / "checkpoints" / "best.pt",
                    cfg,
                    model,
                    epoch,
                    arch,
                    conditioning,
                    provenance,
                    dataset_hash,
                    weights,
                )
            else:
                evaluations_without_improvement += 1
            _save_checkpoint(
                run / "checkpoints" / "last.pt",
                cfg,
                model,
                epoch,
                arch,
                conditioning,
                provenance,
                dataset_hash,
                weights,
            )
            if 0 < cfg.patience <= evaluations_without_improvement:
                stop = True

    _write_history(run / "history.jsonl", history)
    _write_summary(
        run / "summary.json",
        cfg,
        epoch,
        arch,
        conditioning,
        provenance,
        dataset_hash,
        weights,
        metric,
        best,
        last_report,
    )
    if cfg.checkpoint_path:
        destination = Path(cfg.checkpoint_path)
        if destination.exists():
            raise FileExistsError(f"checkpoint destination {destination} already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / "checkpoints" / "last.pt", destination)
    return run


def _require_train_val(
    dests: Sequence[str | Path],
    manifests: Sequence[DatasetManifest],
    kind: str,
) -> None:
    """Require every dataset to carry train and validation rows for the task."""
    for dest, manifest in zip(dests, manifests, strict=True):
        splits = {shard.split for shard in manifest.shards if shard.task == kind}
        missing = {"train", "val"} - splits
        if missing:
            raise ValueError(f"dataset {dest} lacks {sorted(missing)} samples for {kind}")


def _save_checkpoint(
    path: Path,
    cfg: TrainConfig,
    model: nn.Module,
    epoch: int,
    arch: str,
    conditioning: str,
    provenance: dict[str, object],
    dataset_hash: str,
    weights: Sequence[float],
) -> None:
    """Write one checkpoint with the full run metadata."""
    torch.save(
        {
            "kind": cfg.kind,
            "arch": arch,
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "conditioning": conditioning,
            "config": asdict(cfg),
            "provenance": provenance,
            "dataset_weights": list(weights),
            "dataset_hash": dataset_hash,
            "input_height_px": EXPORT_HEIGHT_PX,
            "input_width_px": EXPORT_WIDTH_PX,
        },
        path,
    )


def _write_history(path: Path, history: Sequence[dict[str, object]]) -> None:
    """Write one JSON record per evaluation."""
    with path.open("w") as stream:
        for record in history:
            stream.write(json.dumps(record) + "\n")


def _write_summary(
    path: Path,
    cfg: TrainConfig,
    epoch: int,
    arch: str,
    conditioning: str,
    provenance: dict[str, object],
    dataset_hash: str,
    weights: Sequence[float],
    metric: str,
    best: float,
    evaluation: dict[str, object] | None,
) -> None:
    """Write the run summary: checkpoint metadata plus the last evaluation."""
    path.write_text(
        json.dumps(
            {
                "kind": cfg.kind,
                "arch": arch,
                "epoch": epoch,
                "conditioning": conditioning,
                "config": asdict(cfg),
                "provenance": provenance,
                "dataset_weights": list(weights),
                "dataset_hash": dataset_hash,
                "input_height_px": EXPORT_HEIGHT_PX,
                "input_width_px": EXPORT_WIDTH_PX,
                "val_metric": metric,
                "best_validation": best,
                "evaluation": evaluation,
            },
            indent=2,
        )
        + "\n"
    )
