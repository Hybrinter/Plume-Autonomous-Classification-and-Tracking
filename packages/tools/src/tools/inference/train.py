"""Plain-torch train loop for the classifier and the segmentor.

The loop is SGD + BCEWithLogitsLoss over a frozen TrainConfig. Each run writes
a directory with config.toml, history.csv, last and best checkpoints, and
summary.json. Data is a processed pack with frozen splits, an unsplit disk
adapter, or a synthetic pack. Batches come from ``DataLoader(SplitDataset)``.

Contains:
  - TrainConfig: frozen hyperparameters.
  - load_train_config: dataclass defaults overlaid with an optional TOML file.
  - overlay_train_config: CLI field overlays.
  - config_digest: 8-hex identity of experiment fields.
  - train: run the loop and write a run directory.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from tools.inference.arch.registry import build, resolve_arch
from tools.inference.cost import count_flops, count_params
from tools.inference.data import (
    ProcessedPack,
    SplitDataset,
    load_disk_batch,
    load_processed_pack,
    make_synthetic_pack,
    write_processed_pack,
)
from tools.inference.metrics import classifier_metrics, segmentor_metrics
from tools.inference.split import DatasetMeta, SplitIndex, SplitRecipe

_TRAIN_KINDS = frozenset({"classifier", "segmentor"})
_VAL_METRICS = frozenset({"f1", "mean_iou", "bce"})
_OPTIMIZERS = frozenset({"sgd", "adamw"})
_SCHEDULERS = frozenset({"none", "cosine"})


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Frozen train hyperparameters.

    Defaults match the flight inference contract (4 bands, 256 px) and a short
    SGD schedule. Spatial size is not frozen in the network; it comes from
    these fields.
    """

    kind: str = "segmentor"
    arch: str = ""
    input_height_px: int = 256
    input_width_px: int = 256
    in_channels: int = 4
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    seed: int = 0
    synthetic_samples: int = 4
    data_dir: str = ""
    checkpoint_path: str = ""
    bit_depth: int = 12
    run_dir: str = "artifacts/runs"
    run_id: str = ""
    val_metric: str = ""
    device: str = ""
    overwrite: bool = False
    optimizer: str = "sgd"
    scheduler: str = "none"
    shuffle: bool = False
    pos_weight: float = 0.0
    augment: bool = False


_DIGEST_SKIP = frozenset({"run_dir", "run_id", "checkpoint_path", "overwrite"})


def config_digest(cfg: TrainConfig) -> str:
    """Return an 8-hex digest of the train fields that identify an experiment.

    Args:
        cfg: Frozen train hyperparameters.

    Returns:
        str: First eight hex characters of SHA-256 over the JSON of fields
        other than ``run_dir``, ``run_id``, ``checkpoint_path``, and
        ``overwrite``.
    """
    payload = {
        item.name: getattr(cfg, item.name)
        for item in fields(TrainConfig)
        if item.name not in _DIGEST_SKIP
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def load_train_config(path: str | None = None) -> TrainConfig:
    """Return TrainConfig defaults, overlaid with a TOML file when `path` is set.

    Args:
        path: Optional TOML file. Known keys match TrainConfig field names.

    Returns:
        TrainConfig: Frozen config.

    Raises:
        OSError / tomllib.TOMLDecodeError / KeyError: on a missing or malformed
        file (tools-side engineering check).
        ValueError: If `kind` is not classifier or segmentor.
    """
    cfg = TrainConfig()
    if path is None:
        return cfg
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    valid = {item.name for item in fields(TrainConfig)}
    kind = str(data["kind"]) if "kind" in data else cfg.kind
    if kind not in _TRAIN_KINDS:
        raise ValueError(f"unknown train kind {kind!r}")
    unknown = [key for key in data if key not in valid]
    if unknown:
        raise ValueError(f"unknown train config keys {unknown!r}")
    return TrainConfig(
        kind=kind,
        arch=str(data.get("arch", cfg.arch)),
        input_height_px=int(data.get("input_height_px", cfg.input_height_px)),
        input_width_px=int(data.get("input_width_px", cfg.input_width_px)),
        in_channels=int(data.get("in_channels", cfg.in_channels)),
        epochs=int(data.get("epochs", cfg.epochs)),
        batch_size=int(data.get("batch_size", cfg.batch_size)),
        learning_rate=float(data.get("learning_rate", cfg.learning_rate)),
        momentum=float(data.get("momentum", cfg.momentum)),
        weight_decay=float(data.get("weight_decay", cfg.weight_decay)),
        seed=int(data.get("seed", cfg.seed)),
        synthetic_samples=int(data.get("synthetic_samples", cfg.synthetic_samples)),
        data_dir=str(data.get("data_dir", cfg.data_dir)),
        checkpoint_path=str(data.get("checkpoint_path", cfg.checkpoint_path)),
        bit_depth=int(data.get("bit_depth", cfg.bit_depth)),
        run_dir=str(data.get("run_dir", cfg.run_dir)),
        run_id=str(data.get("run_id", cfg.run_id)),
        val_metric=str(data.get("val_metric", cfg.val_metric)),
        device=str(data.get("device", cfg.device)),
        overwrite=bool(data["overwrite"]) if "overwrite" in data else cfg.overwrite,
        optimizer=str(data.get("optimizer", cfg.optimizer)),
        scheduler=str(data.get("scheduler", cfg.scheduler)),
        shuffle=bool(data["shuffle"]) if "shuffle" in data else cfg.shuffle,
        pos_weight=float(data.get("pos_weight", cfg.pos_weight)),
        augment=bool(data["augment"]) if "augment" in data else cfg.augment,
    )


def overlay_train_config(
    cfg: TrainConfig,
    kind: str | None = None,
    arch: str | None = None,
    data_dir: str | None = None,
    checkpoint_path: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    input_height_px: int | None = None,
    input_width_px: int | None = None,
    seed: int | None = None,
    run_dir: str | None = None,
    run_id: str | None = None,
    in_channels: int | None = None,
    learning_rate: float | None = None,
    momentum: float | None = None,
    weight_decay: float | None = None,
    synthetic_samples: int | None = None,
    bit_depth: int | None = None,
    val_metric: str | None = None,
    device: str | None = None,
    overwrite: bool | None = None,
    optimizer: str | None = None,
    scheduler: str | None = None,
    shuffle: bool | None = None,
    pos_weight: float | None = None,
    augment: bool | None = None,
) -> TrainConfig:
    """Return a copy of `cfg` with any non-None CLI overlays applied.

    Args:
        cfg: Base config from defaults or TOML.
        kind: Optional kind overlay.
        arch: Optional architecture overlay.
        data_dir: Optional disk adapter or processed-pack path.
        checkpoint_path: Optional extra copy of last.pt.
        epochs: Optional epoch count.
        batch_size: Optional batch size.
        input_height_px: Optional height.
        input_width_px: Optional width.
        seed: Optional RNG seed.
        run_dir: Optional parent directory for runs.
        run_id: Optional run directory name.
        in_channels: Optional input band count.
        learning_rate: Optional SGD learning rate.
        momentum: Optional SGD momentum.
        weight_decay: Optional weight decay.
        synthetic_samples: Optional synthetic pack size.
        bit_depth: Optional DN bit depth.
        val_metric: Optional best-checkpoint metric name.
        device: Optional torch device string.
        overwrite: Optional replace-existing-run flag.
        optimizer: Optional ``sgd`` or ``adamw``.
        scheduler: Optional ``none`` or ``cosine``.
        shuffle: Optional train-loader shuffle flag.
        pos_weight: Optional positive-class BCE weight. ``<= 0`` disables.
        augment: Optional train-split flip and rotation flag.

    Returns:
        TrainConfig: Frozen overlay.
    """
    return replace(
        cfg,
        kind=kind if kind is not None else cfg.kind,
        arch=arch if arch is not None else cfg.arch,
        input_height_px=input_height_px if input_height_px is not None else cfg.input_height_px,
        input_width_px=input_width_px if input_width_px is not None else cfg.input_width_px,
        in_channels=in_channels if in_channels is not None else cfg.in_channels,
        epochs=epochs if epochs is not None else cfg.epochs,
        batch_size=batch_size if batch_size is not None else cfg.batch_size,
        learning_rate=learning_rate if learning_rate is not None else cfg.learning_rate,
        momentum=momentum if momentum is not None else cfg.momentum,
        weight_decay=weight_decay if weight_decay is not None else cfg.weight_decay,
        seed=seed if seed is not None else cfg.seed,
        synthetic_samples=(
            synthetic_samples if synthetic_samples is not None else cfg.synthetic_samples
        ),
        data_dir=data_dir if data_dir is not None else cfg.data_dir,
        checkpoint_path=checkpoint_path if checkpoint_path is not None else cfg.checkpoint_path,
        bit_depth=bit_depth if bit_depth is not None else cfg.bit_depth,
        run_dir=run_dir if run_dir is not None else cfg.run_dir,
        run_id=run_id if run_id is not None else cfg.run_id,
        val_metric=val_metric if val_metric is not None else cfg.val_metric,
        device=device if device is not None else cfg.device,
        overwrite=overwrite if overwrite is not None else cfg.overwrite,
        optimizer=optimizer if optimizer is not None else cfg.optimizer,
        scheduler=scheduler if scheduler is not None else cfg.scheduler,
        shuffle=shuffle if shuffle is not None else cfg.shuffle,
        pos_weight=pos_weight if pos_weight is not None else cfg.pos_weight,
        augment=augment if augment is not None else cfg.augment,
    )


def _repo_sha() -> str:
    """Return HEAD SHA, or ``unknown`` when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except OSError:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    sha = result.stdout.strip()
    return sha if sha else "unknown"


def _write_config_toml(path: Path, cfg: TrainConfig) -> None:
    """Write TrainConfig fields as a TOML table."""
    lines: list[str] = []
    for item in fields(TrainConfig):
        value: object = getattr(cfg, item.name)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{item.name} = "{escaped}"')
        elif isinstance(value, bool):
            lines.append(f"{item.name} = {'true' if value else 'false'}")
        else:
            lines.append(f"{item.name} = {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_val_metric(kind: str, name: str) -> str:
    """Return the configured val metric or the kind default."""
    if name:
        if name not in _VAL_METRICS:
            raise ValueError(f"unknown val_metric {name!r}")
        return name
    return "f1" if kind == "classifier" else "mean_iou"


def _pack_from_config(cfg: TrainConfig, run_root: Path) -> ProcessedPack:
    """Load a processed pack, an unsplit disk adapter, or a synthetic pack."""
    if cfg.data_dir:
        root = Path(cfg.data_dir)
        if (root / "splits.json").is_file():
            return load_processed_pack(root, bit_depth=cfg.bit_depth)
        return _unsplit_disk_pack(root, cfg)
    images, masks, labels = make_synthetic_pack(
        n=max(int(cfg.synthetic_samples), 3),
        channels=cfg.in_channels,
        height=cfg.input_height_px,
        width=cfg.input_width_px,
        seed=cfg.seed,
    )
    dest = run_root / "synthetic_pack"
    write_processed_pack(dest, images, masks, labels, SplitRecipe(seed=cfg.seed), source_doi="")
    return load_processed_pack(dest, bit_depth=cfg.bit_depth)


def _unsplit_disk_pack(root: Path, cfg: TrainConfig) -> ProcessedPack:
    """Wrap a labels-or-masks directory as an all-train ProcessedPack."""
    batch = load_disk_batch(root, cfg.kind, bit_depth=cfg.bit_depth)
    n = int(batch.images.shape[0])
    height = int(batch.images.shape[2])
    width = int(batch.images.shape[3])
    if cfg.kind == "segmentor":
        masks = batch.targets
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        labels = (masks.reshape(n, -1).amax(dim=1) > 0.0).to(dtype=torch.float32).reshape(n, 1)
    else:
        labels = batch.targets
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        masks = torch.zeros((n, 1, height, width), dtype=torch.float32)
    splits = SplitIndex(train=tuple(range(n)), val=(), test=())
    meta = DatasetMeta(
        dataset_hash="unsplit",
        source_doi="",
        n=n,
        height=height,
        width=width,
        in_channels=int(batch.images.shape[1]),
    )
    return ProcessedPack(
        images=batch.images,
        masks=masks,
        labels=labels,
        splits=splits,
        meta=meta,
        pack_dir=root,
    )


def _loader(
    pack: ProcessedPack,
    kind: str,
    split: str,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
    augment: bool = False,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Return a deterministic DataLoader over one named split."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        SplitDataset(pack, kind, split, augment=augment, seed=seed),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Return SGD or AdamW from ``cfg.optimizer``."""
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def _make_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Return a cosine scheduler, or None when ``scheduler`` is ``none``."""
    if cfg.scheduler in {"", "none"}:
        return None
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(cfg.epochs), 1))
    raise ValueError(f"unknown scheduler {cfg.scheduler!r}")


def _gather(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run eval-mode inference over a loader and return logits plus targets."""
    model.eval()
    logit_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            logit_chunks.append(logits.detach().cpu())
            target_chunks.append(targets.cpu())
    if not logit_chunks:
        empty = torch.zeros((0, 1), dtype=torch.float32)
        return empty, empty
    return torch.cat(logit_chunks, dim=0), torch.cat(target_chunks, dim=0)


def _score(kind: str, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Return a flat metric dict for one split."""
    if kind == "classifier":
        report = classifier_metrics(logits, targets)
        return {
            "loss": report.bce,
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
        "loss": report_seg.bce,
        "mean_iou": report_seg.mean_iou,
        "mean_dice": report_seg.mean_dice,
        "mean_iou_blob_gate": report_seg.mean_iou_blob_gate,
        "bce": report_seg.bce,
    }


def _val_score(metrics: dict[str, float], name: str) -> float:
    """Return the scalar used for best-checkpoint selection.

    Notes:
        BCE is minimized. Other metrics are maximized. The caller compares with
        that convention.
    """
    return float(metrics[name])


def _is_better(name: str, current: float, best: float) -> bool:
    """Return True when current should replace best."""
    if name == "bce":
        return current < best
    return current > best


def _write_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: TrainConfig,
    arch: str,
    dataset_hash: str,
    epoch: int,
) -> None:
    """Write a checkpoint dict with weights and train identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "kind": cfg.kind,
        "arch": arch,
        "state_dict": model.state_dict(),
        "in_channels": cfg.in_channels,
        "input_height_px": cfg.input_height_px,
        "input_width_px": cfg.input_width_px,
        "dataset_hash": dataset_hash,
        "epoch": epoch,
        "config": asdict(cfg),
    }
    torch.save(payload, path)


def train(config: TrainConfig | None = None) -> Path:
    """Run SGD + BCEWithLogitsLoss and write a run directory.

    Args:
        config: Train hyperparameters. None uses TrainConfig defaults.

    Returns:
        Path: Run directory containing config, history, checkpoints, and summary.

    Raises:
        ValueError: If `config.kind`, architecture, optimizer, or scheduler is
            unknown, or the train split is empty.
        FileExistsError: If the run directory already has ``summary.json`` and
            ``overwrite`` is false.

    Notes:
        `checkpoints/last.pt` updates every epoch. `checkpoints/best.pt` stores
        the best validation score. Classifier default val metric is F1.
        Segmentor default val metric is mean IoU.
    """
    cfg = config if config is not None else TrainConfig()
    if cfg.kind not in _TRAIN_KINDS:
        raise ValueError(f"unknown train kind {cfg.kind!r}")
    if cfg.optimizer not in _OPTIMIZERS:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    if cfg.scheduler not in _SCHEDULERS:
        raise ValueError(f"unknown scheduler {cfg.scheduler!r}")
    arch = resolve_arch(cfg.kind, cfg.arch)
    val_metric = _default_val_metric(cfg.kind, cfg.val_metric)
    run_id = cfg.run_id if cfg.run_id else f"{cfg.kind}-{arch}-{cfg.seed}-{config_digest(cfg)}"
    run_root = Path(cfg.run_dir) / run_id
    if (run_root / "summary.json").is_file() and not cfg.overwrite:
        raise FileExistsError(f"run directory exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
    pack = _pack_from_config(cfg, run_root)
    model = build(cfg.kind, arch, cfg.in_channels)
    n_params = count_params(model)
    flops = count_flops(model, (1, cfg.in_channels, cfg.input_height_px, cfg.input_width_px))
    model.to(device)
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg)
    if cfg.pos_weight > 0.0:
        weight = torch.tensor([cfg.pos_weight], dtype=torch.float32, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    batch = max(int(cfg.batch_size), 1)
    train_idx = pack.splits.train
    val_idx = pack.splits.val
    if not train_idx:
        raise ValueError("train split is empty")
    train_loader = _loader(
        pack,
        cfg.kind,
        "train",
        batch,
        shuffle=cfg.shuffle,
        seed=cfg.seed,
        augment=cfg.augment,
    )
    train_score_loader = _loader(pack, cfg.kind, "train", batch, seed=cfg.seed)
    val_loader = _loader(pack, cfg.kind, "val", batch, seed=cfg.seed) if val_idx else None

    _write_config_toml(run_root / "config.toml", cfg)
    history_path = run_root / "history.csv"
    history_fields: list[str] | None = None
    best_score: float | None = None
    best_epoch = 0

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        rows: list[dict[str, object]] = []
        train_logits, train_targets = _gather(model, train_score_loader, device)
        train_metrics = _score(cfg.kind, train_logits, train_targets)
        rows.append({"epoch": epoch, "split": "train", **train_metrics})
        if val_loader is not None:
            val_logits, val_targets = _gather(model, val_loader, device)
            val_metrics = _score(cfg.kind, val_logits, val_targets)
            rows.append({"epoch": epoch, "split": "val", **val_metrics})
            score = _val_score(val_metrics, val_metric)
        else:
            score = _val_score(train_metrics, val_metric)

        if history_fields is None:
            history_fields = list(rows[0].keys())
            with history_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=history_fields)
                writer.writeheader()
                writer.writerows(rows)
        else:
            with history_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=history_fields)
                writer.writerows(rows)

        last_path = ckpt_dir / "last.pt"
        _write_checkpoint(last_path, model, cfg, arch, pack.meta.dataset_hash, epoch)
        if best_score is None or _is_better(val_metric, score, best_score):
            best_score = score
            best_epoch = epoch
            _write_checkpoint(ckpt_dir / "best.pt", model, cfg, arch, pack.meta.dataset_hash, epoch)

    if cfg.checkpoint_path:
        extra = Path(cfg.checkpoint_path)
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_bytes((ckpt_dir / "last.pt").read_bytes())

    summary = {
        "run_id": run_id,
        "kind": cfg.kind,
        "arch": arch,
        "best_epoch": best_epoch,
        "best_val_metric": best_score,
        "val_metric": val_metric,
        "dataset_hash": pack.meta.dataset_hash,
        "model_repo_sha": _repo_sha(),
        "seed": cfg.seed,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(pack.splits.test),
        "epochs": cfg.epochs,
        "device": device,
        "n_params": n_params,
        "flops": flops,
        "optimizer": cfg.optimizer,
        "scheduler": cfg.scheduler,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_root
