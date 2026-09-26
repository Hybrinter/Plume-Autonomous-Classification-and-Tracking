"""Plain-torch train loop for the classifier and the segmentor.

``canvas is None`` trains on chip batches from a ``DataLoader``. A set
``canvas`` builds flight frames with :func:`tools.ml_models.data.canvas.sample_view`.
Each run writes ``config.toml``, ``history.csv``, ``summary.json``, and
``checkpoints/last.pt`` plus ``checkpoints/best.pt``. Canvas runs also write
``batch_shapes.json``.

Contains:
  - train: run the loop and write a run directory.
  - is_cuda_oom: detect a CUDA allocator failure.
  - next_batch_after_oom: halve a batch size, or raise at size 1.
  - fit_batch_size: lower the batch until one training step fits.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader

from tools.ml_models.arch.registry import build, resolve_arch
from tools.ml_models.data.canvas import CanvasConfig, Chip, sample_view
from tools.ml_models.data.meta import DatasetMeta as MlDatasetMeta
from tools.ml_models.data.pack import ProcessedPack as MlPack
from tools.ml_models.data.pack import load_processed_pack as load_canvas_pack
from tools.ml_models.train.config import (
    TrainConfig,
    apply_train_mapping,
    config_digest,
    resolve_train_channels,
    write_train_config_toml,
)
from tools.ml_models.train.cost import count_flops, count_params
from tools.ml_models.train.losses import LOSS_NAMES, PlumeLoss, build_loss
from tools.ml_models.train.metrics import classifier_metrics, segmentor_metrics
from tools.ml_models.train.recipe import DatasetMeta, SplitIndex, SplitRecipe
from tools.ml_models.train.samples import ProcessedPack as InferencePack
from tools.ml_models.train.samples import SplitDataset, load_disk_batch, make_synthetic_pack
from tools.ml_models.train.samples import load_processed_pack as load_inference_pack
from tools.ml_models.train.samples import write_processed_pack as write_inference_pack

_TRAIN_KINDS = frozenset({"classifier", "segmentor"})
_VAL_METRICS = frozenset({"f1", "mean_iou", "bce"})
_OPTIMIZERS = frozenset({"sgd", "adamw"})
_SCHEDULERS = frozenset({"none", "cosine"})


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


def _validate_train_config(cfg: TrainConfig) -> None:
    """Raise ValueError when a train field is outside the closed sets."""
    if cfg.kind not in _TRAIN_KINDS:
        raise ValueError(f"unknown train kind {cfg.kind!r}")
    if cfg.optimizer not in _OPTIMIZERS:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    if cfg.scheduler not in _SCHEDULERS:
        raise ValueError(f"unknown scheduler {cfg.scheduler!r}")
    if cfg.loss not in LOSS_NAMES:
        raise ValueError(f"unknown loss {cfg.loss!r}")
    if cfg.max_steps is not None and cfg.max_steps < 1:
        raise ValueError(f"max_steps must be >= 1; got {cfg.max_steps}")


def _default_val_metric(kind: str, name: str) -> str:
    """Return the configured val metric or the kind default."""
    if name:
        if name not in _VAL_METRICS:
            raise ValueError(f"unknown val_metric {name!r}")
        return name
    match kind:
        case "classifier":
            return "f1"
        case "segmentor":
            return "mean_iou"
        case _:
            raise ValueError(f"unknown train kind {kind!r}")


def _band_names(meta: object, channels: int) -> tuple[str, ...]:
    """Return pack band names, or ``b0`` .. ``b{C-1}`` when the meta has none."""
    names = getattr(meta, "band_names", None)
    if (
        isinstance(names, tuple)
        and len(names) == channels
        and all(isinstance(item, str) for item in names)
    ):
        return names
    return tuple(f"b{index}" for index in range(channels))


def _provenance(meta: object) -> tuple[str, str]:
    """Return ingest path and radiometry when ``meta`` records them."""
    if isinstance(meta, MlDatasetMeta):
        return meta.ingest_path, meta.radiometry
    return "", ""


def _write_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: TrainConfig,
    arch: str,
    dataset_hash: str,
    epoch: int,
    *,
    band_names: tuple[str, ...],
    in_channels: int,
    ingest_path: str = "",
    radiometry: str = "",
) -> None:
    """Write a checkpoint dict with weights and train identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "kind": cfg.kind,
        "arch": arch,
        "state_dict": model.state_dict(),
        "in_channels": in_channels,
        "input_height_px": cfg.input_height_px,
        "input_width_px": cfg.input_width_px,
        "dataset_hash": dataset_hash,
        "epoch": epoch,
        "band_names": list(band_names),
        "config": asdict(cfg),
    }
    if ingest_path:
        payload["ingest_path"] = ingest_path
    if radiometry:
        payload["radiometry"] = radiometry
    canvas = cfg.canvas
    if canvas is not None:
        payload["frame_hw"] = (int(canvas.frame_hw[0]), int(canvas.frame_hw[1]))
        payload["window_px"] = int(canvas.window_px)
    torch.save(payload, path)


def _pack_from_config(cfg: TrainConfig, run_root: Path) -> InferencePack:
    """Load a processed pack, an unsplit disk adapter, or a synthetic pack."""
    if cfg.data_dir:
        root = Path(cfg.data_dir)
        if (root / "splits.json").is_file():
            return load_inference_pack(
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
    write_inference_pack(dest, images, masks, labels, SplitRecipe(seed=cfg.seed), source_doi="")
    return load_inference_pack(dest, bit_depth=cfg.bit_depth)


def _unsplit_disk_pack(root: Path, cfg: TrainConfig) -> InferencePack:
    """Wrap a labels-or-masks directory as an all-train ProcessedPack."""
    batch = load_disk_batch(root, cfg.kind, bit_depth=cfg.bit_depth)
    n = int(batch.images.shape[0])
    height = int(batch.images.shape[2])
    width = int(batch.images.shape[3])
    match cfg.kind:
        case "segmentor":
            masks = batch.targets
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            labels = (masks.reshape(n, -1).amax(dim=1) > 0.0).to(dtype=torch.float32).reshape(n, 1)
        case "classifier":
            labels = batch.targets
            if labels.ndim == 1:
                labels = labels.reshape(-1, 1)
            masks = torch.zeros((n, 1, height, width), dtype=torch.float32)
        case _:
            raise ValueError(f"unknown train kind {cfg.kind!r}")
    splits = SplitIndex(train=tuple(range(n)), val=(), test=())
    meta = DatasetMeta(
        dataset_hash="unsplit",
        source_doi="",
        n=n,
        height=height,
        width=width,
        in_channels=int(batch.images.shape[1]),
    )
    return InferencePack(
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
    pack: InferencePack,
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
    match cfg.optimizer:
        case "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        case "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        case _:
            raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def _make_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Return a cosine scheduler, or None when ``scheduler`` is ``none``."""
    match cfg.scheduler:
        case "" | "none":
            return None
        case "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(int(cfg.epochs), 1)
            )
        case _:
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
    match kind:
        case "classifier":
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
        case "segmentor":
            report_seg = segmentor_metrics(logits, targets)
            return {
                "loss": report_seg.bce,
                "mean_iou": report_seg.mean_iou,
                "mean_dice": report_seg.mean_dice,
                "mean_iou_blob_gate": report_seg.mean_iou_blob_gate,
                "bce": report_seg.bce,
            }
        case _:
            raise ValueError(f"unknown train kind {kind!r}")


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


def _append_history(
    path: Path,
    rows: list[dict[str, object]],
    header: list[str] | None,
) -> list[str]:
    """Append history rows and return the column names."""
    fields = list(rows[0].keys()) if header is None else header
    mode = "w" if header is None else "a"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if header is None:
            writer.writeheader()
        writer.writerows(rows)
    return fields


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


def _prepare_run(cfg: TrainConfig, arch: str) -> tuple[str, Path]:
    """Return the run id and directory, creating the checkpoint folder."""
    run_id = cfg.run_id if cfg.run_id else f"{cfg.kind}-{arch}-{cfg.seed}-{config_digest(cfg)}"
    run_root = Path(cfg.run_dir) / run_id
    if (run_root / "summary.json").is_file() and not cfg.overwrite:
        raise FileExistsError(f"run directory exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_id, run_root


def _device_of(cfg: TrainConfig) -> str:
    """Return the configured device, or CUDA when it is present."""
    if cfg.device:
        return cfg.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _copy_extra_checkpoint(cfg: TrainConfig, last_path: Path) -> None:
    """Copy ``last.pt`` to ``checkpoint_path`` when that field is set."""
    if not cfg.checkpoint_path:
        return
    extra = Path(cfg.checkpoint_path)
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(last_path.read_bytes())


def _train_chips(cfg: TrainConfig, arch: str) -> Path:
    """Train on chip batches and return the run directory."""
    val_metric = _default_val_metric(cfg.kind, cfg.val_metric)
    disk_pack: InferencePack | None = None
    if cfg.data_dir:
        disk_pack = _pack_from_config(cfg, Path(cfg.run_dir))
        resolve_train_channels(cfg, disk_pack)
    run_id, run_root = _prepare_run(cfg, arch)
    ckpt_dir = run_root / "checkpoints"

    torch.manual_seed(cfg.seed)
    device = _device_of(cfg)
    torch.backends.cudnn.benchmark = device.startswith("cuda")
    pack = disk_pack if disk_pack is not None else _pack_from_config(cfg, run_root)
    if disk_pack is None:
        resolve_train_channels(cfg, pack)
    names = _band_names(pack.meta, cfg.in_channels)
    ingest_path, radiometry = _provenance(pack.meta)
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

    write_train_config_toml(run_root / "config.toml", written)
    history_path = run_root / "history.csv"
    history_fields: list[str] | None = None
    best_score: float | None = None
    best_epoch = 0
    stale_epochs = 0
    stopped_early = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    eval_interval = max(int(cfg.eval_interval), 1)
    started_at = time.perf_counter()
    steps_done = 0
    hit_limit = False

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
            steps_done += 1
            if cfg.max_steps is not None and steps_done >= int(cfg.max_steps):
                hit_limit = True
                break
        if scheduler is not None:
            scheduler.step()
        if epoch % eval_interval != 0 and epoch != int(cfg.epochs) and not hit_limit:
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

        history_fields = _append_history(history_path, rows, history_fields)
        last_path = ckpt_dir / "last.pt"
        _write_checkpoint(
            last_path,
            model,
            cfg,
            arch,
            pack.meta.dataset_hash,
            epoch,
            band_names=names,
            in_channels=cfg.in_channels,
            ingest_path=ingest_path,
            radiometry=radiometry,
        )
        if best_score is None or _is_better(val_metric, score, best_score):
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            _write_checkpoint(
                ckpt_dir / "best.pt",
                model,
                cfg,
                arch,
                pack.meta.dataset_hash,
                epoch,
                band_names=names,
                in_channels=cfg.in_channels,
                ingest_path=ingest_path,
                radiometry=radiometry,
            )
        else:
            stale_epochs += 1
            if int(cfg.patience) > 0 and stale_epochs >= int(cfg.patience):
                stopped_early = True
                break
        if hit_limit or stopped_early:
            break

    train_seconds = time.perf_counter() - started_at
    _copy_extra_checkpoint(cfg, ckpt_dir / "last.pt")
    summary: dict[str, object] = {
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
        "in_channels": cfg.in_channels,
        "band_names": list(names),
        "input_height_px": int(cfg.input_height_px),
        "input_width_px": int(cfg.input_width_px),
    }
    if ingest_path:
        summary["ingest_path"] = ingest_path
    if radiometry:
        summary["radiometry"] = radiometry
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_root


def _chips_from_pack(pack: MlPack, split: str) -> tuple[Chip, ...]:
    """Return one chip per row of ``split``.

    A row is annotated when its mask has a positive pixel. A positive label
    with an all-zero mask stays unannotated.
    """
    chips: list[Chip] = []
    for index in pack.splits.for_name(split):
        image = np.array(pack.images[index], dtype=np.float32, copy=True)
        mask = np.array(pack.masks[index], dtype=np.float32, copy=True)
        label = float(np.asarray(pack.labels[index], dtype=np.float32).reshape(-1)[0])
        chips.append(
            Chip(
                image=image,
                mask=mask,
                label=label,
                group_id=str(index),
                split=split,
                annotated=bool(np.any(mask > 0.0)),
            )
        )
    return tuple(chips)


def _scene_config(canvas: CanvasConfig, chips: Sequence[Chip]) -> CanvasConfig:
    """Return ``canvas`` with paste disabled when the split has no polygon."""
    if any(chip.annotated for chip in chips):
        return canvas
    if canvas.max_plumes == 0:
        return canvas
    return replace(canvas, max_plumes=0)


def _canvas_batch(
    chips: Sequence[Chip],
    canvas: CanvasConfig,
    rng: np.random.Generator,
    *,
    full_frame: bool,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack ``count`` views that share ``full_frame``."""
    scene = _scene_config(canvas, chips)
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[float] = []
    for _ in range(count):
        sample = sample_view(chips, scene, rng, full_frame=full_frame)
        images.append(sample.image)
        masks.append(sample.mask)
        labels.append(float(sample.label))
    image_t = torch.from_numpy(np.stack(images))
    mask_t = torch.from_numpy(np.stack(masks))
    label_t = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
    return image_t, mask_t, label_t


def _forward_tensor(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return the tensor from ``model(images)``."""
    output = model(images)
    if not isinstance(output, torch.Tensor):
        raise TypeError("model forward must return a tensor")
    return output


def _spatial_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor | None:
    """Return ``spatial(images)`` when the module defines that method."""
    method = getattr(model, "spatial", None)
    if not callable(method):
        return None
    mapped = method(images)
    if not isinstance(mapped, torch.Tensor):
        raise TypeError("spatial() must return a tensor")
    return mapped


def _max_logit(logits: torch.Tensor) -> torch.Tensor:
    """Return one logit per sample, reducing spatial axes with amax."""
    if logits.ndim <= 2:
        return logits.reshape(-1, 1)
    reduced = logits.amax(dim=tuple(range(2, logits.ndim)))
    return reduced.reshape(-1, 1)


def _classifier_outputs(
    model: nn.Module, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the max logit and the spatial map when the module has one."""
    spatial = _spatial_logits(model, images)
    if spatial is not None:
        return _max_logit(spatial), spatial
    return _max_logit(_forward_tensor(model, images)), None


def _pool_mask_any(mask: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Pool a mask so a cell is positive when any input pixel in it is positive.

    Args:
        mask: Float mask ``(N, 1, H, W)``.
        out_hw: Spatial size of the logit map.

    Returns:
        torch.Tensor: Float ``(N, 1, h, w)`` in {0, 1}.
    """
    binary = (mask > 0).to(dtype=torch.float32)
    pooled: torch.Tensor = functional.adaptive_max_pool2d(binary, out_hw)
    return pooled


def _backward(loss: torch.Tensor) -> None:
    """Run backward. ``Tensor.backward`` has no parameter annotations."""
    loss.backward()  # type: ignore[no-untyped-call]


def _scaled_backward(scaler: torch.amp.GradScaler, loss: torch.Tensor) -> None:
    """Scale ``loss`` and run backward."""
    scaled = scaler.scale(loss)
    scaled.backward()  # type: ignore[no-untyped-call]


def _as_loss(value: object) -> torch.Tensor:
    """Return ``value`` when it is a tensor."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("loss must be a tensor")
    return value


def _canvas_loss(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    *,
    kind: str,
    focal: PlumeLoss,
    pos_weight: float,
) -> torch.Tensor:
    """Return the canvas objective for one batch.

    Segmentors use focal Dice on the full-resolution logits. Classifiers use
    BCE-with-logits on the max logit. When the module has ``spatial`` and a
    sample mask contains a polygon, focal Dice on that map is added. The mask
    is pooled onto the spatial grid first.
    """
    if kind == "segmentor":
        return _as_loss(focal(_forward_tensor(model, images), masks))
    logits, spatial = _classifier_outputs(model, images)
    weight: torch.Tensor | None = None
    if pos_weight > 0.0:
        weight = torch.tensor([pos_weight], dtype=logits.dtype, device=logits.device)
    bce = functional.binary_cross_entropy_with_logits(
        logits,
        labels.to(dtype=logits.dtype),
        pos_weight=weight,
    )
    if spatial is None:
        return bce
    pasted = masks.flatten(start_dim=2).amax(dim=2).reshape(-1) > 0
    if not bool(pasted.any().item()):
        return bce
    pooled = _pool_mask_any(masks, (int(spatial.shape[-2]), int(spatial.shape[-1])))
    aux = _as_loss(focal(spatial[pasted], pooled[pasted]))
    return bce + aux


def _canvas_metric_name(kind: str) -> str:
    """Return the full-frame metric used to pick ``best.pt``."""
    match kind:
        case "classifier":
            return "f1"
        case "segmentor":
            return "mean_dice"
        case _:
            raise ValueError(f"unknown train kind {kind!r}")


def _canvas_loss_name(kind: str, model: nn.Module) -> str:
    """Return the summary loss name for a canvas run."""
    if kind == "segmentor":
        return "focal_dice"
    if callable(getattr(model, "spatial", None)):
        return "bce+focal_dice"
    return "bce"


def _gather_canvas(
    model: nn.Module,
    chips: Sequence[Chip],
    canvas: CanvasConfig,
    rng: np.random.Generator,
    *,
    kind: str,
    batch: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score ``len(chips)`` full frames. Windows are not included."""
    if not chips:
        empty = torch.zeros((0, 1), dtype=torch.float32)
        return empty, empty
    model.eval()
    logit_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    remaining = len(chips)
    with torch.no_grad():
        while remaining > 0:
            take = min(batch, remaining)
            images, masks, labels = _canvas_batch(chips, canvas, rng, full_frame=True, count=take)
            images = images.to(device)
            if kind == "classifier":
                logits, _spatial = _classifier_outputs(model, images)
                targets = labels
            else:
                logits = _forward_tensor(model, images)
                targets = masks
            logit_chunks.append(logits.detach().cpu())
            target_chunks.append(targets.detach().cpu())
            remaining -= take
    return torch.cat(logit_chunks, dim=0), torch.cat(target_chunks, dim=0)


def _steps_per_epoch(n_chips: int, batch: int) -> int:
    """Return how many optimizer steps cover the chip list once."""
    return max(1, (max(n_chips, 1) + batch - 1) // batch)


def _train_canvas(cfg: TrainConfig, arch: str) -> Path:
    """Train on flight-frame views and return the run directory."""
    canvas = cfg.canvas
    if canvas is None:
        raise ValueError("canvas training requires TrainConfig.canvas")
    if not cfg.data_dir:
        raise ValueError("canvas training requires data_dir")
    pack = load_canvas_pack(cfg.data_dir)
    channels = resolve_train_channels(cfg, pack)
    if int(pack.meta.height) != canvas.chip_side or int(pack.meta.width) != canvas.chip_side:
        raise ValueError(
            f"pack spatial size {pack.meta.height}x{pack.meta.width} "
            f"does not match canvas chip_side {canvas.chip_side}"
        )
    run_id, run_root = _prepare_run(cfg, arch)
    ckpt_dir = run_root / "checkpoints"
    names = _band_names(pack.meta, channels)
    ingest_path, radiometry = _provenance(pack.meta)

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(canvas.seed)
    torch.backends.cudnn.benchmark = False
    device = _device_of(cfg)
    use_amp = device.startswith("cuda")
    frame_h, frame_w = canvas.frame_hw
    cost_model = build(cfg.kind, arch, channels)
    n_params = count_params(cost_model)
    flops = count_flops(cost_model, (1, channels, int(frame_h), int(frame_w)))
    del cost_model

    train_chips = _chips_from_pack(pack, "train")
    val_chips = _chips_from_pack(pack, "val")
    if not train_chips:
        raise ValueError("train split is empty")

    def _warmup(candidate: int) -> None:
        probe = build(cfg.kind, arch, channels).to(device)
        try:
            opt = _make_optimizer(probe, cfg)
            focal = build_loss(
                "focal_dice",
                focal_gamma=cfg.focal_gamma,
                focal_alpha=cfg.focal_alpha,
            ).to(device)
            images = torch.zeros((candidate, channels, int(frame_h), int(frame_w)), device=device)
            masks = torch.zeros((candidate, 1, int(frame_h), int(frame_w)), device=device)
            labels = torch.zeros((candidate, 1), device=device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    loss = _canvas_loss(
                        probe,
                        images,
                        masks,
                        labels,
                        kind=cfg.kind,
                        focal=focal,
                        pos_weight=cfg.pos_weight,
                    )
                probe_scaler = torch.amp.GradScaler("cuda")
                _scaled_backward(probe_scaler, loss)
            else:
                loss = _canvas_loss(
                    probe,
                    images,
                    masks,
                    labels,
                    kind=cfg.kind,
                    focal=focal,
                    pos_weight=cfg.pos_weight,
                )
                _backward(loss)
        finally:
            del probe
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    batch = fit_batch_size(int(cfg.batch_size), _warmup)
    written = apply_train_mapping(cfg, {"batch_size": batch}) if batch != cfg.batch_size else cfg
    torch.manual_seed(cfg.seed)
    model = build(cfg.kind, arch, channels)
    model.to(device)
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg)
    focal = build_loss(
        "focal_dice",
        focal_gamma=cfg.focal_gamma,
        focal_alpha=cfg.focal_alpha,
    ).to(device)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    write_train_config_toml(run_root / "config.toml", written)

    per_epoch = _steps_per_epoch(len(train_chips), batch)
    planned = per_epoch * int(cfg.epochs)
    if cfg.max_steps is not None:
        planned = min(planned, int(cfg.max_steps))
    if planned < 1:
        raise ValueError("canvas schedule has no optimizer steps")

    val_metric = _canvas_metric_name(cfg.kind)
    history_path = run_root / "history.csv"
    history_fields: list[str] | None = None
    best_score: float | None = None
    best_epoch = 0
    stale_epochs = 0
    stopped_early = False
    batch_shapes: list[list[int]] = []
    eval_interval = max(int(cfg.eval_interval), 1)
    started_at = time.perf_counter()
    scheduler_epoch = 0

    for step in range(planned):
        model.train()
        full_frame = step % int(canvas.full_frame_every) == 0
        images, masks, labels = _canvas_batch(
            train_chips,
            canvas,
            rng,
            full_frame=full_frame,
            count=batch,
        )
        batch_shapes.append([int(dim) for dim in images.shape])
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda"):
                loss = _canvas_loss(
                    model,
                    images,
                    masks,
                    labels,
                    kind=cfg.kind,
                    focal=focal,
                    pos_weight=cfg.pos_weight,
                )
            if scaler is None:
                raise RuntimeError("CUDA autocast requires a GradScaler")
            _scaled_backward(scaler, loss)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = _canvas_loss(
                model,
                images,
                masks,
                labels,
                kind=cfg.kind,
                focal=focal,
                pos_weight=cfg.pos_weight,
            )
            _backward(loss)
            optimizer.step()

        epoch = step // per_epoch + 1
        epoch_end = (step + 1) == min(planned, epoch * per_epoch)
        if epoch_end and scheduler is not None and scheduler_epoch < epoch:
            scheduler.step()
            scheduler_epoch = epoch
        if not epoch_end:
            continue
        if epoch % eval_interval != 0 and step + 1 != planned:
            continue

        rows: list[dict[str, object]] = []
        train_logits, train_targets = _gather_canvas(
            model,
            train_chips,
            canvas,
            rng,
            kind=cfg.kind,
            batch=batch,
            device=device,
        )
        train_metrics = _score(cfg.kind, train_logits, train_targets)
        rows.append({"epoch": epoch, "split": "train", **train_metrics})
        if val_chips:
            val_logits, val_targets = _gather_canvas(
                model,
                val_chips,
                canvas,
                rng,
                kind=cfg.kind,
                batch=batch,
                device=device,
            )
            val_metrics = _score(cfg.kind, val_logits, val_targets)
            rows.append({"epoch": epoch, "split": "val", **val_metrics})
            score = _val_score(val_metrics, val_metric)
        else:
            score = _val_score(train_metrics, val_metric)

        history_fields = _append_history(history_path, rows, history_fields)
        last_path = ckpt_dir / "last.pt"
        _write_checkpoint(
            last_path,
            model,
            cfg,
            arch,
            pack.meta.dataset_hash,
            epoch,
            band_names=names,
            in_channels=channels,
            ingest_path=ingest_path,
            radiometry=radiometry,
        )
        if best_score is None or _is_better(val_metric, score, best_score):
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            _write_checkpoint(
                ckpt_dir / "best.pt",
                model,
                cfg,
                arch,
                pack.meta.dataset_hash,
                epoch,
                band_names=names,
                in_channels=channels,
                ingest_path=ingest_path,
                radiometry=radiometry,
            )
        else:
            stale_epochs += 1
            if int(cfg.patience) > 0 and stale_epochs >= int(cfg.patience):
                stopped_early = True
                break

    train_seconds = time.perf_counter() - started_at
    _copy_extra_checkpoint(cfg, ckpt_dir / "last.pt")
    (run_root / "batch_shapes.json").write_text(
        json.dumps(batch_shapes) + "\n",
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "run_id": run_id,
        "kind": cfg.kind,
        "arch": arch,
        "best_epoch": best_epoch,
        "best_val_metric": best_score,
        "val_metric": val_metric,
        "dataset_hash": pack.meta.dataset_hash,
        "model_repo_sha": _repo_sha(),
        "seed": cfg.seed,
        "n_train": len(train_chips),
        "n_val": len(val_chips),
        "n_test": len(pack.splits.test),
        "epochs": cfg.epochs,
        "device": device,
        "n_params": n_params,
        "flops": flops,
        "optimizer": cfg.optimizer,
        "scheduler": cfg.scheduler,
        "loss": _canvas_loss_name(cfg.kind, model),
        "amp": use_amp,
        "batch_size": batch,
        "stopped_early": stopped_early,
        "train_seconds": round(train_seconds, 3),
        "frame_hw": [int(frame_h), int(frame_w)],
        "window_px": int(canvas.window_px),
        "in_channels": channels,
        "band_names": list(names),
        "ingest_path": ingest_path,
        "radiometry": radiometry,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_root


def train(config: TrainConfig | None = None) -> Path:
    """Run the train loop and write a run directory.

    Args:
        config: Train hyperparameters. None uses TrainConfig defaults.

    Returns:
        Path: Run directory containing config, history, checkpoints, and summary.

    Raises:
        ValueError: If `config.kind`, architecture, optimizer, scheduler, or
            loss is unknown, the train split is empty, or a canvas pack
            disagrees with ``in_channels`` or ``chip_side``.
        FileExistsError: If the run directory already has ``summary.json`` and
            ``overwrite`` is false.

    Notes:
        ``checkpoints/last.pt`` updates every scored epoch.
        ``checkpoints/best.pt`` stores the best validation score. With
        ``canvas is None``, the classifier default is F1 and the segmentor
        default is mean IoU. With a canvas, selection uses full-frame
        classifier F1 or segmentor Dice. The test split is not scored.
        ``max_steps`` stops the optimizer early. ``None`` runs full epochs.
        A CUDA out-of-memory error on the first probe halves ``batch_size``.
    """
    cfg = config if config is not None else TrainConfig()
    _validate_train_config(cfg)
    arch = resolve_arch(cfg.kind, cfg.arch)
    if cfg.canvas is not None:
        return _train_canvas(cfg, arch)
    return _train_chips(cfg, arch)
