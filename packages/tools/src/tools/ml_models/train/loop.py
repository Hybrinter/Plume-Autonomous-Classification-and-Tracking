"""Imperative training loop for GSD-conditioned models over finished datasets.

A run validates the dataset manifest, measured provenance, and train/val
availability before reserving an exclusive run directory. The RUNNING
``execution.json`` lands immediately after reservation, then every attempted
batch and epoch is appended to ``history.jsonl`` and flushed; execution
settles COMPLETED, FAILED, or INTERRUPTED. Checkpoints serialize at evaluated
epochs; best keeps the best measured validation score, last the most recent
evaluated state, and the selected one may be published to
``cfg.checkpoint_path``. Test split tensors are never loaded; provenance still
traverses test row metadata for integrity.

Contains:
  - train: the public ``Result`` boundary.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import IO, TYPE_CHECKING, Protocol, cast

from flight.libs.types import Err, Ok, Result
from pydantic import TypeAdapter

from tools.ml_models.analysis.artifacts import checksum_file
from tools.ml_models.analysis.config import EvaluationConfig
from tools.ml_models.analysis.contracts import CheckpointIdentity, SplitEvidence
from tools.ml_models.analysis.training import (
    CheckpointRecord,
    EpochRecord,
    EvaluationRecord,
    RunStatus,
    StepRecord,
    TrainingExecution,
    is_improvement,
    observe_loss,
    reduce_epoch,
    selected_metric,
)
from tools.ml_models.dataset.manifest import DatasetManifest, load_manifest
from tools.ml_models.train.config import (
    TrainConfig,
    config_digest,
    validation_metric,
    write_train_config_toml,
)
from tools.ml_models.train.provenance import training_provenance

if TYPE_CHECKING:
    import torch
    from torch import nn

    from tools.ml_models.analysis.capture import CaptureSink
    from tools.ml_models.analysis.cost import ResourceEvidence
    from tools.ml_models.dataset.loader import Batch
    from tools.ml_models.train.losses import LossComponents, PlumeLoss


class _EvaluateSplit(Protocol):
    """The ``evaluate_split`` boundary invoked once per evaluated epoch."""

    def __call__(
        self,
        model: nn.Module,
        dataset: Path,
        manifest: DatasetManifest,
        cfg: EvaluationConfig,
        objective: PlumeLoss | None = None,
        capture: CaptureSink | None = None,
    ) -> Result[SplitEvidence, str]:
        """Score one split exhaustively and return split evidence."""
        ...


class _MakeLoader(Protocol):
    """The ``make_loader`` boundary yielding seeded single-shard batches."""

    def __call__(
        self,
        dataset: str | Path,
        task: str,
        split: str,
        batch_size: int,
        seed: int,
        *,
        n_batches: int,
    ) -> Iterator[Batch]:
        """Yield ``n_batches`` seeded ``(image, gsd, target)`` batches."""
        ...


class _MeasureResources(Protocol):
    """The ``measure_resources`` boundary over one model and input shape."""

    def __call__(
        self, model: nn.Module, input_shape: tuple[int, ...]
    ) -> Result[ResourceEvidence, str]:
        """Return partial resource evidence or an explicit error."""
        ...


_EXECUTION = TypeAdapter(TrainingExecution)
_EVIDENCE = TypeAdapter(SplitEvidence)
_RECORD: TypeAdapter[StepRecord | EpochRecord | EvaluationRecord] = TypeAdapter(
    StepRecord | EpochRecord | EvaluationRecord
)

_UNSAFE_RUN_ID = ("/", "\\", ":")


def train(cfg: TrainConfig | None = None) -> Result[Path, str]:
    """Train one configured run and return its run directory.

    Args:
        cfg: Training configuration. None uses :class:`TrainConfig` defaults.

    Returns:
        Result[Path, str]: The owned run directory on success, or an explicit
        error. Failed runs keep their earlier history, checkpoints, and a
        FAILED or INTERRUPTED ``execution.json`` rather than being erased.
    """
    resolved = cfg if cfg is not None else TrainConfig()
    try:
        return _train(resolved)
    except KeyboardInterrupt:
        return Err("training interrupted")
    except (OSError, ValueError, RuntimeError, TypeError, ImportError, OverflowError) as exc:
        return Err(str(exc))


def _train(cfg: TrainConfig) -> Result[Path, str]:
    """Validate inputs, then delegate to the execution shell."""
    if not cfg.dataset.strip():
        return Err("train requires exactly one finished dataset")
    run_id = cfg.run_id or f"{cfg.kind}-{config_digest(cfg)}-{time.time_ns()}"
    if (
        not run_id
        or run_id != run_id.strip()
        or run_id in (".", "..")
        or any(mark in run_id for mark in _UNSAFE_RUN_ID)
        or Path(run_id).name != run_id
    ):
        return Err(f"run_id must be one safe path component; got {run_id!r}")
    dataset = Path(cfg.dataset)
    try:
        manifest = load_manifest(dataset / "dataset.json")
        provenance = training_provenance(dataset, manifest, cfg.kind)
        _require_train_val(cfg.dataset, manifest, cfg.kind)
    except (OSError, ValueError, RuntimeError) as exc:
        return Err(str(exc))
    dataset_root = dataset.resolve()
    run = Path(cfg.run_dir) / run_id
    if run.resolve().is_relative_to(dataset_root):
        return Err(f"run directory {run} lies inside the dataset root")
    destination = Path(cfg.checkpoint_path) if cfg.checkpoint_path.strip() else None
    if destination is not None:
        if destination.resolve().is_relative_to(dataset_root):
            return Err(f"checkpoint destination {destination} lies inside the dataset root")
        if os.path.lexists(destination):
            return Err(f"checkpoint destination {destination} already exists")
    return _execute(cfg, dataset, manifest, provenance, run, destination)


def _require_train_val(dataset: str, manifest: DatasetManifest, kind: str) -> None:
    """Require the dataset to carry train and validation rows for the task."""
    splits = {shard.split for shard in manifest.shards if shard.task == kind}
    missing = {"train", "val"} - splits
    if missing:
        raise ValueError(f"dataset {dataset} lacks {sorted(missing)} samples for {kind}")


def _private_temp(directory: Path, name: str) -> tuple[int, Path]:
    """Create a unique private temp file beside ``name`` and return fd, path."""
    fd, raw = tempfile.mkstemp(dir=directory, prefix=f".{name}.", suffix=".part")
    return fd, Path(raw)


def _atomic_write(path: Path, payload: bytes) -> None:
    """Replace ``path`` with ``payload`` through a private temp file."""
    fd, temporary = _private_temp(path.parent, path.name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    """Serialize one checkpoint through a private temp file."""
    import torch

    fd, temporary = _private_temp(path.parent, path.name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_file(source: Path, destination: Path) -> None:
    """Byte-copy ``source`` to ``destination`` through a private temp file."""
    fd, temporary = _private_temp(destination.parent, destination.name)
    try:
        with os.fdopen(fd, "wb") as handle:
            with source.open("rb") as reader:
                shutil.copyfileobj(reader, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_checkpoint(source: Path, destination: Path) -> Result[None, str]:
    """Publish a byte copy of ``source`` at ``destination`` exclusively.

    The copy lands in a private temp inside the destination directory, then
    ``os.link`` publishes it atomically only when no destination exists, so a
    racing or preexisting destination can never be overwritten.
    """
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = _private_temp(destination.parent, destination.name)
    except OSError as exc:
        return Err(f"cannot prepare checkpoint destination {destination}: {exc}")
    try:
        with os.fdopen(fd, "wb") as handle:
            with source.open("rb") as reader:
                shutil.copyfileobj(reader, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    except OSError as exc:
        return Err(f"cannot publish checkpoint to {destination}: {exc}")
    finally:
        temporary.unlink(missing_ok=True)
    return Ok(None)


def _gradient_norm(model: nn.Module) -> float:
    """L2 norm over all gradient-bearing parameters, in float64."""
    return math.sqrt(
        sum(
            float((parameter.grad.detach().double() ** 2).sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )


def _gradients_finite(model: nn.Module) -> bool:
    """Check every produced parameter gradient for finite values."""
    import torch

    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def _synchronize(device: str) -> None:
    """Fence the configured CUDA device at timing boundaries; no-op on CPU."""
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(torch.device(device))


class _RunState:
    """Owned run directory state: counters, warnings, and the history handle."""

    def __init__(
        self,
        run: Path,
        execution: TrainingExecution,
        *,
        arch: str,
        conditioning: str,
        provenance: dict[str, object],
        metric: str,
    ) -> None:
        self.run = run
        self.execution = execution
        self.arch = arch
        self.conditioning = conditioning
        self.provenance = provenance
        self.metric = metric
        self.warnings = list(execution.warnings)
        self.epoch = 0
        self.batch = 0
        self.optimizer_step = 0
        self.samples_seen = 0
        self.best_value: float | None = None
        self.best: CheckpointRecord | None = None
        self.last: CheckpointRecord | None = None
        self.nonimproving = 0
        self.stop_reason: str | None = None
        self._history: IO[bytes] | None = None

    def open_history(self) -> None:
        """Open the append handle for ``history.jsonl``."""
        self._history = (self.run / "history.jsonl").open("ab")

    def append(self, record: StepRecord | EpochRecord | EvaluationRecord) -> None:
        """Append one durable history record and flush it to disk."""
        if self._history is None:
            raise RuntimeError("history handle is not open")
        self._history.write(_RECORD.dump_json(record) + b"\n")
        self._history.flush()
        os.fsync(self._history.fileno())

    def _selected(self) -> CheckpointRecord | None:
        """Resolve the selection policy against current checkpoint records."""
        return self.best if self.execution.selection == "best" else self.last

    def _trusted(self) -> tuple[CheckpointRecord | None, CheckpointRecord | None]:
        """Drop any recorded checkpoint whose on-disk bytes lost their hash."""
        trusted: list[CheckpointRecord | None] = []
        for record in (self.best, self.last):
            if record is None:
                trusted.append(None)
                continue
            sha = checksum_file(self.run / record.path)
            if isinstance(sha, Err) or sha.value != record.identity.sha256:
                self.warnings.append(
                    f"{record.path} bytes no longer match the recorded "
                    "checkpoint hash; dropping it from the terminal execution."
                )
                trusted.append(None)
            else:
                trusted.append(record)
        return trusted[0], trusted[1]

    def update(self) -> None:
        """Atomically rewrite ``execution.json`` with current RUNNING progress."""
        self.execution = replace(
            self.execution,
            epoch=self.epoch,
            optimizer_step=self.optimizer_step,
            samples_seen=self.samples_seen,
            best=self.best,
            last=self.last,
            selected=self._selected(),
            warnings=tuple(self.warnings),
        )
        _atomic_write(self.run / "execution.json", _EXECUTION.dump_json(self.execution))

    def complete(self) -> None:
        """Atomically persist the COMPLETED terminal state."""
        self.execution = replace(
            self.execution,
            status="COMPLETED",
            stop_reason=self.stop_reason,
            epoch=self.epoch,
            optimizer_step=self.optimizer_step,
            samples_seen=self.samples_seen,
            best=self.best,
            last=self.last,
            selected=self._selected(),
            warnings=tuple(self.warnings),
        )
        _atomic_write(self.run / "execution.json", _EXECUTION.dump_json(self.execution))

    def fail(self, status: RunStatus, message: str) -> Result[Path, str]:
        """Persist a terminal status and return an error naming the run."""
        best, last = self._trusted()
        selected = best if self.execution.selection == "best" else last
        try:
            self.execution = replace(
                self.execution,
                status=status,
                stop_reason=message,
                epoch=self.epoch,
                optimizer_step=self.optimizer_step,
                samples_seen=self.samples_seen,
                best=best,
                last=last,
                selected=selected,
                warnings=tuple(self.warnings),
            )
            _atomic_write(self.run / "execution.json", _EXECUTION.dump_json(self.execution))
        except (OSError, ValueError, RuntimeError) as exc:
            return Err(f"{message}; could not persist {status} status: {exc}; run {self.run}")
        return Err(f"{message}; run {self.run}")

    def close(self) -> None:
        """Release the history handle if it was opened."""
        if self._history is not None:
            self._history.close()


def _execute(
    cfg: TrainConfig,
    dataset: Path,
    manifest: DatasetManifest,
    provenance: dict[str, object],
    run: Path,
    destination: Path | None,
) -> Result[Path, str]:
    """Build the training objects, reserve the run, and drive the loop."""
    import torch

    from tools.ml_models.analysis import evaluate as evaluator
    from tools.ml_models.analysis.cost import measure_resources
    from tools.ml_models.arch.film import (
        CONDITIONING_ID,
        IGNORED_CONDITIONING_ID,
        IgnoreGsd,
    )
    from tools.ml_models.arch.registry import build as build_model
    from tools.ml_models.arch.registry import resolve_arch
    from tools.ml_models.dataset.loader import make_loader
    from tools.ml_models.train.losses import build_loss

    if os.path.lexists(run):
        return Err(f"run directory {run} already exists")
    try:
        torch.manual_seed(cfg.seed)
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        arch = resolve_arch(cfg.kind, cfg.arch)
        model = build_model(cfg.kind, cfg.arch, cast(int, provenance["in_channels"]))
        model.to(device)
        conditioning = IGNORED_CONDITIONING_ID if isinstance(model, IgnoreGsd) else CONDITIONING_ID
        metric = validation_metric(cfg.kind, cfg.val_metric)
        optimizer: torch.optim.Optimizer
        if cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
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
        criterion.to(device)
        warnings: list[str] = []
        amp_enabled = cfg.amp and device.startswith("cuda")
        if cfg.amp and not amp_enabled:
            warnings.append(
                "cfg.amp requested but the device is not CUDA; batches run in full precision."
            )
        scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
        execution = TrainingExecution(
            kind=cfg.kind,
            arch=arch,
            dataset_hash=manifest.dataset_hash,
            conditioning=conditioning,
            config=asdict(cfg),
            provenance=dict(provenance),
            metric=metric,
            selection=cfg.selected_checkpoint,
            amp_enabled=amp_enabled,
            warnings=tuple(warnings),
        )
        state = _RunState(
            run,
            execution,
            arch=arch,
            conditioning=conditioning,
            provenance=provenance,
            metric=metric,
        )
        run.mkdir(parents=True)
    except FileExistsError:
        return Err(f"run directory {run} already exists")
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        return Err(f"{exc}; run {run}")
    except KeyboardInterrupt:
        return Err(f"training interrupted; run {run}")

    try:
        state.update()
        (run / "checkpoints").mkdir()
        write_train_config_toml(run / "config.toml", cfg)
        state.open_history()
        _loop(
            state,
            cfg,
            dataset,
            manifest,
            model,
            optimizer,
            scheduler,
            criterion,
            scaler,
            evaluator.evaluate_split,
            make_loader,
            measure_resources,
            device,
        )
        if destination is not None:
            selected = state.execution.selected
            if selected is None:
                raise ValueError("no evaluated checkpoint exists to publish")
            published = _publish_checkpoint(state.run / selected.path, destination)
            if isinstance(published, Err):
                raise ValueError(published.error)
        state.complete()
        return Ok(run)
    except KeyboardInterrupt:
        return state.fail("INTERRUPTED", "training interrupted")
    except (OSError, ValueError, RuntimeError, TypeError, OverflowError) as exc:
        return state.fail("FAILED", str(exc))
    finally:
        state.close()


def _loop(
    state: _RunState,
    cfg: TrainConfig,
    dataset: Path,
    manifest: DatasetManifest,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    criterion: PlumeLoss,
    scaler: torch.amp.GradScaler | None,
    evaluate_split: _EvaluateSplit,
    make_loader: _MakeLoader,
    measure_resources: _MeasureResources,
    device: str,
) -> None:
    """Drive epochs, records, and evaluation until a terminal condition."""
    while state.epoch < cfg.epochs and state.stop_reason is None:
        state.epoch += 1
        model.train()
        n_batches = math.ceil(cast(int, state.provenance["train_samples"]) / cfg.batch_size)
        loader = make_loader(
            cfg.dataset,
            cfg.kind,
            "train",
            cfg.batch_size,
            cfg.seed + state.epoch,
            n_batches=n_batches,
        )
        steps: list[StepRecord] = []
        for images, gsd, targets in loader:
            record = _batch(
                state,
                cfg,
                model,
                optimizer,
                criterion,
                scaler,
                device,
                images,
                gsd,
                targets,
            )
            steps.append(record)
            state.append(record)
            state.update()
            if cfg.max_steps is not None and state.optimizer_step >= cfg.max_steps:
                state.stop_reason = "max_steps"
                break
        reduced = reduce_epoch(tuple(steps))
        if isinstance(reduced, Err):
            raise ValueError(reduced.error)
        state.append(reduced.value)
        state.update()
        if scheduler is not None and reduced.value.n_updates > 0:
            scheduler.step()
        if (
            state.epoch % cfg.eval_interval == 0
            or state.epoch == cfg.epochs
            or state.stop_reason is not None
        ):
            _evaluate_and_checkpoint(
                state, cfg, dataset, manifest, model, criterion, evaluate_split, device
            )
            if state.stop_reason is None and 0 < cfg.patience <= state.nonimproving:
                state.stop_reason = "early_stopping"
    if state.stop_reason is None:
        state.stop_reason = "epochs"
    _write_resources(state, cfg, model, measure_resources)


def _batch(
    state: _RunState,
    cfg: TrainConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: PlumeLoss,
    scaler: torch.amp.GradScaler | None,
    device: str,
    images: torch.Tensor,
    gsd: torch.Tensor,
    targets: torch.Tensor,
) -> StepRecord:
    """Run one attempted batch and return its durable step record."""
    import torch

    _synchronize(device)
    start = time.perf_counter()
    learning_rates = tuple(float(group["lr"]) for group in optimizer.param_groups)
    images = images.to(device)
    gsd = gsd.to(device)
    targets = targets.to(device)
    n_samples = int(targets.shape[0])
    if not bool(((targets == 0.0) | (targets == 1.0)).all()):
        raise ValueError("batch targets must be exact binary values")
    optimizer.zero_grad()
    update_applied = True
    gradient_norm: float | None = None
    scale_before: float | None = None
    scale_after: float | None = None
    if scaler is not None:
        with torch.autocast("cuda"):
            output = model(images, gsd)
            if not isinstance(output, torch.Tensor) or output.shape != targets.shape:
                raise ValueError(f"model output does not match target shape {tuple(targets.shape)}")
            if not bool(torch.isfinite(output).all()):
                raise ValueError("model output is nonfinite")
            components: LossComponents = criterion.per_sample_components(output, targets)
        observed = observe_loss(components, criterion.pixel_weight, criterion.dice_weight)
        if isinstance(observed, Err):
            raise ValueError(observed.error)
        loss = components.total.mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite objective before optimizer update")
        before = float(scaler.get_scale())
        cast(Callable[[], None], scaler.scale(loss).backward)()
        scaler.unscale_(optimizer)
        finite = _gradients_finite(model)
        if finite and cfg.gradient_diagnostics:
            gradient_norm = _gradient_norm(model)
        scaler.step(optimizer)
        scaler.update()
        after = float(scaler.get_scale())
        if not finite:
            if after >= before:
                raise RuntimeError("AMP scaler applied an update over non-finite gradients")
            update_applied = False
            state.warnings.append(f"AMP skipped the optimizer update at batch {state.batch + 1}.")
        else:
            update_applied = after >= before
            if not update_applied:
                state.warnings.append(
                    f"AMP skipped the optimizer update at batch {state.batch + 1}."
                )
        if cfg.amp_diagnostics:
            scale_before = before
            scale_after = after
    else:
        output = model(images, gsd)
        if not isinstance(output, torch.Tensor) or output.shape != targets.shape:
            raise ValueError(f"model output does not match target shape {tuple(targets.shape)}")
        if not bool(torch.isfinite(output).all()):
            raise ValueError("model output is nonfinite")
        components = criterion.per_sample_components(output, targets)
        observed = observe_loss(components, criterion.pixel_weight, criterion.dice_weight)
        if isinstance(observed, Err):
            raise ValueError(observed.error)
        loss = components.total.mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite objective before optimizer update")
        cast(Callable[[], None], loss.backward)()
        if not _gradients_finite(model):
            raise RuntimeError("non-finite gradients before optimizer update")
        if cfg.gradient_diagnostics:
            gradient_norm = _gradient_norm(model)
        optimizer.step()
    _synchronize(device)
    duration = max(time.perf_counter() - start, 1e-9)
    state.batch += 1
    state.samples_seen += n_samples
    if update_applied:
        state.optimizer_step += 1
    return StepRecord(
        epoch=state.epoch,
        batch=state.batch,
        optimizer_step=state.optimizer_step,
        samples_seen=state.samples_seen,
        n_samples=n_samples,
        loss=observed.value,
        learning_rates=learning_rates,
        duration_s=duration,
        throughput_images_s=n_samples / duration,
        update_applied=update_applied,
        gradient_norm=gradient_norm,
        amp_scale_before=scale_before,
        amp_scale_after=scale_after,
    )


def _evaluate_and_checkpoint(
    state: _RunState,
    cfg: TrainConfig,
    dataset: Path,
    manifest: DatasetManifest,
    model: nn.Module,
    criterion: PlumeLoss,
    evaluate_split: _EvaluateSplit,
    device: str,
) -> None:
    """Evaluate train/val, serialize last.pt, and update best/selected."""
    _synchronize(device)
    start = time.perf_counter()
    train_result = evaluate_split(
        model,
        dataset,
        manifest,
        EvaluationConfig(kind=cfg.kind, split="train", batch_size=cfg.batch_size, device=device),
        objective=criterion,
    )
    if isinstance(train_result, Err):
        raise ValueError(f"train evaluation failed: {train_result.error}")
    val_result = evaluate_split(
        model,
        dataset,
        manifest,
        EvaluationConfig(kind=cfg.kind, split="val", batch_size=cfg.batch_size, device=device),
        objective=criterion,
    )
    if isinstance(val_result, Err):
        raise ValueError(f"validation evaluation failed: {val_result.error}")
    _synchronize(device)
    duration = max(time.perf_counter() - start, 1e-9)
    train_evidence = replace(train_result.value, curves=(), strata=())
    val_evidence = replace(val_result.value, curves=(), strata=())
    measured = selected_metric(val_evidence, state.metric)
    if isinstance(measured, Err):
        raise ValueError(measured.error)
    value, direction = measured.value
    _save_checkpoint(
        state.run / "checkpoints" / "last.pt",
        {
            "kind": cfg.kind,
            "arch": state.arch,
            "state_dict": model.state_dict(),
            "epoch": state.epoch,
            "conditioning": state.conditioning,
            "config": asdict(cfg),
            "provenance": state.provenance,
            "dataset_hash": manifest.dataset_hash,
            "step": state.optimizer_step,
            "samples_seen": state.samples_seen,
            "metric": state.metric,
            "direction": direction,
            "value": value,
            "validation": _EVIDENCE.dump_python(val_evidence, mode="json"),
        },
    )
    sha = checksum_file(state.run / "checkpoints" / "last.pt")
    if isinstance(sha, Err):
        raise ValueError(f"cannot checksum checkpoint: {sha.error}")
    identity = CheckpointIdentity(
        sha256=sha.value,
        kind=cfg.kind,
        arch=state.arch,
        training_dataset_hash=manifest.dataset_hash,
        epoch=state.epoch,
        step=state.optimizer_step,
    )
    bound_train = replace(train_evidence, checkpoint_hash=sha.value)
    bound_val = replace(val_evidence, checkpoint_hash=sha.value)
    record = CheckpointRecord(
        path="checkpoints/last.pt",
        identity=identity,
        metric=state.metric,
        direction=direction,
        value=value,
        validation=bound_val,
    )
    state.last = record
    improved = is_improvement(value, state.best_value, direction)
    if improved:
        _copy_file(
            state.run / "checkpoints" / "last.pt",
            state.run / "checkpoints" / "best.pt",
        )
        state.best = replace(record, path="checkpoints/best.pt")
        state.best_value = value
        state.nonimproving = 0
    else:
        state.nonimproving += 1
    state.append(
        EvaluationRecord(
            epoch=state.epoch,
            optimizer_step=state.optimizer_step,
            samples_seen=state.samples_seen,
            train=bound_train,
            validation=bound_val,
            checkpoint=record,
            improved=improved,
            duration_s=duration,
        )
    )
    state.update()


def _write_resources(
    state: _RunState,
    cfg: TrainConfig,
    model: nn.Module,
    measure_resources: _MeasureResources,
) -> None:
    """Persist per-shape partial resource evidence or an explicit error."""
    channels = cast(int, state.provenance["in_channels"])
    entries: list[dict[str, object]] = []
    for shape in cast(list[list[int]], state.provenance["spatial_shapes"]):
        height, width = int(shape[0]), int(shape[1])
        for batch in sorted({1, cfg.batch_size}):
            entry: dict[str, object] = {
                "image_shape": [batch, channels, height, width],
            }
            measured = measure_resources(model, (batch, channels, height, width))
            if isinstance(measured, Ok):
                entry["evidence"] = asdict(measured.value)
            else:
                entry["error"] = measured.error
            entries.append(entry)
    _atomic_write(
        state.run / "resources.json",
        (
            json.dumps(
                {
                    "records": entries,
                    "checkpoint": (asdict(state.last.identity) if state.last is not None else None),
                    "model_state": "last_evaluated_checkpoint",
                }
            )
            + "\n"
        ).encode(),
    )
