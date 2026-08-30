"""Plain-torch train loop for the classifier and the segmentor.

The loop runs a frozen TrainConfig: SGD or AdamW against one of the objectives
in :mod:`tools.inference.losses`, optionally under CUDA mixed precision. Each
run writes a directory with config.toml, history.csv, last and best
checkpoints, and summary.json. Data is a processed pack with frozen splits, an
unsplit disk adapter, or a synthetic pack. Batches come from
``DataLoader(SplitDataset)``.

Two knobs exist for long searches rather than for model quality. ``patience``
abandons a run once the validation metric stops improving, and
``eval_interval`` scores less often than every epoch; scoring walks both splits
in full, so it can cost about as much as training the epoch did.

Contains:
  - TrainConfig: frozen hyperparameters.
  - load_train_config: dataclass defaults overlaid with an optional TOML file.
  - overlay_train_config: CLI field overlays.
  - apply_train_mapping: overlay from a string-key mapping.
  - config_digest: 8-hex identity of experiment fields.
  - train: run the loop and write a run directory.
  - is_cuda_oom: detect a CUDA allocator failure.
  - next_batch_after_oom: halve a batch size, or raise at size 1.
  - fit_batch_size: lower the batch until one training step fits.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
import tomllib
from collections.abc import Callable
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass
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
from tools.inference.losses import (
    DEFAULT_FOCAL_ALPHA,
    DEFAULT_FOCAL_GAMMA,
    LOSS_NAMES,
    LossName,
    build_loss,
)
from tools.inference.metrics import classifier_metrics, segmentor_metrics
from tools.inference.split import DatasetMeta, SplitIndex, SplitRecipe

TrainKind = Literal["classifier", "segmentor"]
OptimizerName = Literal["sgd", "adamw"]
SchedulerName = Literal["none", "cosine"]
_TRAIN_KINDS = frozenset({"classifier", "segmentor"})
_VAL_METRICS = frozenset({"f1", "mean_iou", "bce"})
_OPTIMIZERS = frozenset({"sgd", "adamw"})
_SCHEDULERS = frozenset({"none", "cosine"})
_SCHEMA = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class TrainConfig:
    """Frozen train hyperparameters.

    Defaults match the flight inference contract (4 bands, 256 px) and a short
    SGD schedule. Spatial size is not frozen in the network; it comes from
    these fields.
    """

    kind: TrainKind = "segmentor"
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
    optimizer: OptimizerName = "sgd"
    scheduler: SchedulerName = "none"
    shuffle: bool = False
    pos_weight: float = 0.0
    augment: bool = False
    loss: LossName = "bce"
    focal_gamma: float = DEFAULT_FOCAL_GAMMA
    focal_alpha: float = DEFAULT_FOCAL_ALPHA
    amp: bool = False
    patience: int = 0
    eval_interval: int = 1


_TRAIN_ADAPTER = TypeAdapter(TrainConfig)


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
        OSError / tomllib.TOMLDecodeError: on a missing or malformed file
        (tools-side engineering check).
        ValidationError: If a key is unknown or a field fails the schema.
    """
    cfg = TrainConfig()
    if path is None:
        return cfg
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    payload: dict[str, object] = dict(data)
    return apply_train_mapping(cfg, payload)


def apply_train_mapping(cfg: TrainConfig, data: dict[str, object]) -> TrainConfig:
    """Return ``cfg`` overlaid with known TrainConfig keys in ``data``.

    Args:
        cfg: Base config.
        data: Mapping of field names to TOML or JSON values.

    Returns:
        TrainConfig: Frozen overlay.

    Raises:
        ValidationError: If a key is unknown or a field fails the schema.
    """
    merged: dict[str, object] = {item.name: getattr(cfg, item.name) for item in fields(TrainConfig)}
    merged.update(data)
    return _TRAIN_ADAPTER.validate_python(merged)


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
    loss: str | None = None,
    focal_gamma: float | None = None,
    focal_alpha: float | None = None,
    amp: bool | None = None,
    patience: int | None = None,
    eval_interval: int | None = None,
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
        loss: Optional objective name from ``tools.inference.losses``.
        focal_gamma: Optional focal focusing exponent.
        focal_alpha: Optional focal positive-class weight.
        amp: Optional CUDA mixed-precision flag.
        patience: Optional early-stop patience in scored epochs. ``<= 0``
            disables.
        eval_interval: Optional epochs between scoring passes.

    Returns:
        TrainConfig: Frozen overlay.
    """
    updates: dict[str, object] = {}
    if kind is not None:
        updates["kind"] = kind
    if arch is not None:
        updates["arch"] = arch
    if data_dir is not None:
        updates["data_dir"] = data_dir
    if checkpoint_path is not None:
        updates["checkpoint_path"] = checkpoint_path
    if epochs is not None:
        updates["epochs"] = epochs
    if batch_size is not None:
        updates["batch_size"] = batch_size
    if input_height_px is not None:
        updates["input_height_px"] = input_height_px
    if input_width_px is not None:
        updates["input_width_px"] = input_width_px
    if seed is not None:
        updates["seed"] = seed
    if run_dir is not None:
        updates["run_dir"] = run_dir
    if run_id is not None:
        updates["run_id"] = run_id
    if in_channels is not None:
        updates["in_channels"] = in_channels
    if learning_rate is not None:
        updates["learning_rate"] = learning_rate
    if momentum is not None:
        updates["momentum"] = momentum
    if weight_decay is not None:
        updates["weight_decay"] = weight_decay
    if synthetic_samples is not None:
        updates["synthetic_samples"] = synthetic_samples
    if bit_depth is not None:
        updates["bit_depth"] = bit_depth
    if val_metric is not None:
        updates["val_metric"] = val_metric
    if device is not None:
        updates["device"] = device
    if overwrite is not None:
        updates["overwrite"] = overwrite
    if optimizer is not None:
        updates["optimizer"] = optimizer
    if scheduler is not None:
        updates["scheduler"] = scheduler
    if shuffle is not None:
        updates["shuffle"] = shuffle
    if pos_weight is not None:
        updates["pos_weight"] = pos_weight
    if augment is not None:
        updates["augment"] = augment
    if loss is not None:
        updates["loss"] = loss
    if focal_gamma is not None:
        updates["focal_gamma"] = focal_gamma
    if focal_alpha is not None:
        updates["focal_alpha"] = focal_alpha
    if amp is not None:
        updates["amp"] = amp
    if patience is not None:
        updates["patience"] = patience
    if eval_interval is not None:
        updates["eval_interval"] = eval_interval
    return apply_train_mapping(cfg, updates) if updates else cfg


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
            return load_processed_pack(
                root, bit_depth=cfg.bit_depth, load_masks=cfg.kind != "classifier"
            )
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


def _loader_for(
    dataset: SplitDataset,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Return a deterministic DataLoader over an already-built split dataset."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
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
    dataset = SplitDataset(pack, kind, split, augment=augment, seed=seed)
    return _loader_for(dataset, batch_size, shuffle=shuffle, seed=seed)


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


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True when ``exc`` is a CUDA allocator failure.

    Args:
        exc: Exception raised during a CUDA step.

    Returns:
        bool: True for ``torch.cuda.OutOfMemoryError`` and for a
            ``RuntimeError`` whose message names out of memory.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and "out of memory" in text


def next_batch_after_oom(batch: int) -> int:
    """Return ``batch // 2``, or raise when the size is already 1.

    Args:
        batch: Batch size that just failed to fit.

    Returns:
        int: Next smaller size, at least 1.

    Raises:
        RuntimeError: If ``batch`` is 1 or less.
    """
    if batch <= 1:
        raise RuntimeError("CUDA out of memory at batch_size=1")
    return max(batch // 2, 1)


def fit_batch_size(requested: int, attempt: Callable[[int], None]) -> int:
    """Return a batch size at or below ``requested`` that ``attempt`` accepts.

    Args:
        requested: Requested batch size. Values below 1 are treated as 1.
        attempt: Called with the candidate size. Must raise a CUDA
            out-of-memory error when that size does not fit.

    Returns:
        int: The first size ``attempt`` accepts, walking ``requested``,
            ``requested // 2``, and so on down to 1.

    Raises:
        RuntimeError: If size 1 still raises CUDA OOM.
        BaseException: Any non-OOM exception from ``attempt``.
    """
    batch = max(int(requested), 1)
    while True:
        try:
            attempt(batch)
            return batch
        except BaseException as exc:
            if not is_cuda_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch = next_batch_after_oom(batch)


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
        Segmentor default val metric is mean IoU. A CUDA out-of-memory error on
        the first step halves ``batch_size`` and retries down to 1. The written
        ``config.toml`` stores the size that fitted, so later eval uses it.
    """
    cfg = config if config is not None else TrainConfig()
    if cfg.kind not in _TRAIN_KINDS:
        raise ValueError(f"unknown train kind {cfg.kind!r}")
    if cfg.optimizer not in _OPTIMIZERS:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    if cfg.scheduler not in _SCHEDULERS:
        raise ValueError(f"unknown scheduler {cfg.scheduler!r}")
    if cfg.loss not in LOSS_NAMES:
        raise ValueError(f"unknown loss {cfg.loss!r}")
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
    # Every batch has the same shape, so letting cuDNN benchmark once and reuse
    # the winning algorithm pays for itself across a multi-epoch run.
    torch.backends.cudnn.benchmark = device.startswith("cuda")
    pack = _pack_from_config(cfg, run_root)
    cost_model = build(cfg.kind, arch, cfg.in_channels)
    n_params = count_params(cost_model)
    flops = count_flops(cost_model, (1, cfg.in_channels, cfg.input_height_px, cfg.input_width_px))
    del cost_model
    train_idx = pack.splits.train
    val_idx = pack.splits.val
    if not train_idx:
        raise ValueError("train split is empty")
    train_dataset = SplitDataset(pack, cfg.kind, "train", augment=cfg.augment, seed=cfg.seed)
    use_amp = bool(cfg.amp) and device.startswith("cuda")

    def _warmup(candidate: int) -> None:
        probe = build(cfg.kind, arch, cfg.in_channels).to(device)
        try:
            opt = _make_optimizer(probe, cfg)
            criterion = build_loss(
                cfg.loss,
                pos_weight=cfg.pos_weight,
                focal_gamma=cfg.focal_gamma,
                focal_alpha=cfg.focal_alpha,
            ).to(device)
            loader = _loader_for(train_dataset, candidate, shuffle=False, seed=cfg.seed)
            batch_x, batch_y = next(iter(loader))
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(probe(batch_x), batch_y)
            if use_amp:
                probe_scaler = torch.amp.GradScaler("cuda", enabled=True)
                probe_scaler.scale(loss).backward()
            else:
                loss.backward()
        finally:
            del probe
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    batch = fit_batch_size(int(cfg.batch_size), _warmup)
    written = apply_train_mapping(cfg, {"batch_size": batch}) if batch != cfg.batch_size else cfg
    torch.manual_seed(cfg.seed)
    model = build(cfg.kind, arch, cfg.in_channels)
    model.to(device)
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg)
    loss_fn = build_loss(
        cfg.loss,
        pos_weight=cfg.pos_weight,
        focal_gamma=cfg.focal_gamma,
        focal_alpha=cfg.focal_alpha,
    ).to(device)
    train_loader = _loader_for(train_dataset, batch, shuffle=cfg.shuffle, seed=cfg.seed)
    train_score_loader = _loader(pack, cfg.kind, "train", batch, seed=cfg.seed)
    val_loader = _loader(pack, cfg.kind, "val", batch, seed=cfg.seed) if val_idx else None

    _write_config_toml(run_root / "config.toml", written)
    history_path = run_root / "history.csv"
    history_fields: list[str] | None = None
    best_score: float | None = None
    best_epoch = 0
    stale_epochs = 0
    stopped_early = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    eval_interval = max(int(cfg.eval_interval), 1)
    started_at = time.perf_counter()

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        train_dataset.set_epoch(epoch)
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()
        # Scoring runs both splits in full, so it can cost as much as the epoch
        # itself. The final epoch is always scored so a run never ends unscored.
        if epoch % eval_interval != 0 and epoch != int(cfg.epochs):
            continue

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
            stale_epochs = 0
            _write_checkpoint(ckpt_dir / "best.pt", model, cfg, arch, pack.meta.dataset_hash, epoch)
        else:
            stale_epochs += 1
            if int(cfg.patience) > 0 and stale_epochs >= int(cfg.patience):
                stopped_early = True
                break

    train_seconds = time.perf_counter() - started_at
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
        "loss": cfg.loss,
        "amp": use_amp,
        "batch_size": batch,
        "stopped_early": stopped_early,
        "train_seconds": round(train_seconds, 3),
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_root
