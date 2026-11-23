"""Strict incremental training records and inference-free history analysis.

Step losses are equal-image means of configured weighted objective terms.
Epoch reductions weight those means by the actual number of sampled images,
including AMP-skipped batches; optimizer_step counts only applied updates.
Evaluation records cover canonical train and exhaustive validation, never test.
An unterminated final JSONL record is ignored only for incomplete executions.
Complete invalid records and nonmonotonic identities are always errors.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math
from dataclasses import field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from flight.libs.types import Err, Ok, Result
from pydantic import ConfigDict, StrictBool, StrictInt, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass

from tools.ml_models.analysis.contracts import (
    CheckpointIdentity,
    FiniteNumber,
    SplitEvidence,
    Task,
    is_sha256,
)
from tools.ml_models.analysis.metrics.definitions import metric_definition

if TYPE_CHECKING:
    from tools.ml_models.train.losses import LossComponents

_SCHEMA = ConfigDict(extra="forbid")
type Direction = Literal["MINIMIZE", "MAXIMIZE"]
type RunStatus = Literal["RUNNING", "COMPLETED", "FAILED", "INTERRUPTED"]


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class LossObservation:
    """Equal-image configured objective and weighted active component means."""

    total: FiniteNumber
    bce: FiniteNumber | None = None
    focal: FiniteNumber | None = None
    dice: FiniteNumber | None = None

    @model_validator(mode="after")
    def _bounds(self) -> LossObservation:
        terms = [value for value in (self.bce, self.focal, self.dice) if value is not None]
        if not terms or any(value < 0 for value in terms) or self.total < 0:
            raise ValueError("objective needs nonnegative active components")
        if not math.isclose(self.total, math.fsum(terms), rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError("weighted objective components do not sum to total")
        return self


def observe_loss(
    components: LossComponents, pixel_weight: float, dice_weight: float
) -> Result[LossObservation, str]:
    """Reduce finite per-image terms in float64 without changing optimizer tensors."""
    try:
        import torch

        n = components.total.numel()
        if components.total.shape != (n,) or n < 1:
            return Err("loss components must be nonempty per-image vectors")
        values: dict[str, float | None] = {}
        for name, tensor, weight in (
            ("total", components.total, 1.0),
            ("bce", components.bce, pixel_weight),
            ("focal", components.focal, pixel_weight),
            ("dice", components.dice, dice_weight),
        ):
            if tensor is None:
                values[name] = None
            elif tensor.shape != (n,) or not bool(torch.isfinite(tensor).all()):
                return Err("loss components are nonfinite or misaligned")
            else:
                values[name] = float(tensor.detach().double().mean().item()) * weight
        return Ok(TypeAdapter(LossObservation).validate_python(values))
    except (ValueError, TypeError, RuntimeError, OverflowError) as exc:
        return Err(f"invalid objective observation: {exc}")


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class StepRecord:
    """One attempted batch with cumulative successful updates and sample exposure."""

    epoch: StrictInt
    batch: StrictInt
    optimizer_step: StrictInt
    samples_seen: StrictInt
    n_samples: StrictInt
    loss: LossObservation
    learning_rates: tuple[FiniteNumber, ...]
    duration_s: FiniteNumber
    throughput_images_s: FiniteNumber
    update_applied: StrictBool = True
    gradient_norm: FiniteNumber | None = None
    amp_scale_before: FiniteNumber | None = None
    amp_scale_after: FiniteNumber | None = None
    schema_version: Literal[1] = 1
    record_kind: Literal["STEP"] = "STEP"

    @model_validator(mode="after")
    def _bounds(self) -> StepRecord:
        if min(self.epoch, self.batch, self.n_samples) < 1 or self.optimizer_step < 0:
            raise ValueError("invalid step counters")
        if self.samples_seen < self.n_samples or self.duration_s <= 0:
            raise ValueError("invalid sample exposure or duration")
        if not math.isclose(
            self.throughput_images_s, self.n_samples / self.duration_s, rel_tol=1e-12
        ):
            raise ValueError("throughput disagrees with samples and duration")
        if not self.learning_rates or any(value < 0 for value in self.learning_rates):
            raise ValueError("learning rates must be nonnegative")
        if self.gradient_norm is not None and self.gradient_norm < 0:
            raise ValueError("gradient norm must be nonnegative")
        if (self.amp_scale_before is None) != (self.amp_scale_after is None):
            raise ValueError("AMP scales must be supplied together")
        if any(
            value is not None and value <= 0
            for value in (self.amp_scale_before, self.amp_scale_after)
        ):
            raise ValueError("AMP scales must be positive")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class EpochRecord:
    """Sample-weighted optimization evidence for a possibly truncated epoch."""

    epoch: StrictInt
    optimizer_step: StrictInt
    samples_seen: StrictInt
    n_samples: StrictInt
    n_batches: StrictInt
    n_updates: StrictInt
    loss: LossObservation
    duration_s: FiniteNumber
    throughput_images_s: FiniteNumber
    learning_rates: tuple[FiniteNumber, ...]
    schema_version: Literal[1] = 1
    record_kind: Literal["EPOCH"] = "EPOCH"

    @model_validator(mode="after")
    def _bounds(self) -> EpochRecord:
        if min(self.epoch, self.n_samples, self.n_batches) < 1:
            raise ValueError("invalid epoch counters")
        if not 0 <= self.n_updates <= self.n_batches or self.optimizer_step < self.n_updates:
            raise ValueError("invalid optimizer update count")
        if self.samples_seen < self.n_samples or self.duration_s <= 0:
            raise ValueError("invalid epoch sample exposure or duration")
        if not math.isclose(
            self.throughput_images_s, self.n_samples / self.duration_s, rel_tol=1e-12
        ):
            raise ValueError("throughput disagrees with samples and duration")
        if not self.learning_rates or any(value < 0 for value in self.learning_rates):
            raise ValueError("invalid learning rates")
        return self


def reduce_epoch(steps: tuple[StepRecord, ...]) -> Result[EpochRecord, str]:
    """Combine actual batch support; never average unequal batch means equally."""
    if not steps or len({step.epoch for step in steps}) != 1:
        return Err("epoch reduction requires records from one nonempty epoch")
    n = sum(step.n_samples for step in steps)
    values: dict[str, float | None] = {}
    for name, terms in (
        ("total", [step.loss.total for step in steps]),
        ("bce", [step.loss.bce for step in steps]),
        ("focal", [step.loss.focal for step in steps]),
        ("dice", [step.loss.dice for step in steps]),
    ):
        if all(term is None for term in terms):
            values[name] = None
        elif any(term is None for term in terms):
            return Err("objective component activity changed inside an epoch")
        else:
            values[name] = math.fsum(
                cast(float, term) * (step.n_samples / n)
                for term, step in zip(terms, steps, strict=True)
            )
    duration = math.fsum(step.duration_s for step in steps)
    try:
        return Ok(
            EpochRecord(
                epoch=steps[-1].epoch,
                optimizer_step=steps[-1].optimizer_step,
                samples_seen=steps[-1].samples_seen,
                n_samples=n,
                n_batches=len(steps),
                n_updates=sum(step.update_applied for step in steps),
                loss=TypeAdapter(LossObservation).validate_python(values),
                duration_s=duration,
                throughput_images_s=n / duration,
                learning_rates=steps[-1].learning_rates,
            )
        )
    except (ValueError, OverflowError) as exc:
        return Err(f"invalid epoch reduction: {exc}")


def selected_metric(evidence: SplitEvidence, name: str) -> Result[tuple[float, Direction], str]:
    """Require an available validation score and its declarative improvement direction."""
    if evidence.split != "val":
        return Err("checkpoint selection requires validation evidence")
    definition = metric_definition(name)
    if isinstance(definition, Err):
        return definition
    direction = definition.value.direction
    if direction == "DESCRIPTIVE":
        return Err(f"metric {name!r} has no improvement direction")
    for metric in evidence.metrics:
        if metric.name == name:
            if metric.value is None:
                return Err(f"selected validation metric {name!r} is unavailable: {metric.reason}")
            return Ok((metric.value, direction))
    return Err(f"selected validation metric {name!r} is absent")


def is_improvement(value: float, best: float | None, direction: Direction) -> bool:
    """Choose strict improvements; ties preserve the earlier best checkpoint."""
    return best is None or (value < best if direction == "MINIMIZE" else value > best)


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class CheckpointRecord:
    """Actual serialized checkpoint identity and validation evidence at that state."""

    path: str
    identity: CheckpointIdentity
    metric: str
    direction: Direction
    value: FiniteNumber
    validation: SplitEvidence

    @model_validator(mode="after")
    def _bounds(self) -> CheckpointRecord:
        if self.path not in ("checkpoints/best.pt", "checkpoints/last.pt"):
            raise ValueError("checkpoint path must name best.pt or last.pt")
        if self.identity.epoch is None or self.identity.step is None:
            raise ValueError("training checkpoint epoch and step are required")
        if self.validation.task != self.identity.kind:
            raise ValueError("checkpoint and validation task disagree")
        if self.validation.dataset_hash != self.identity.training_dataset_hash:
            raise ValueError("checkpoint and validation dataset disagree")
        if self.validation.checkpoint_hash != self.identity.sha256:
            raise ValueError("validation checkpoint hash disagrees")
        measured = selected_metric(self.validation, self.metric)
        if isinstance(measured, Err):
            raise ValueError(measured.error)
        if measured.value != (self.value, self.direction):
            raise ValueError("checkpoint selected score disagrees with validation evidence")
        return self


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class EvaluationRecord:
    """Canonical train/validation scores at one serialized last checkpoint."""

    epoch: StrictInt
    optimizer_step: StrictInt
    samples_seen: StrictInt
    train: SplitEvidence
    validation: SplitEvidence
    checkpoint: CheckpointRecord
    improved: StrictBool
    duration_s: FiniteNumber
    schema_version: Literal[1] = 1
    record_kind: Literal["EVALUATION"] = "EVALUATION"

    @model_validator(mode="after")
    def _bounds(self) -> EvaluationRecord:
        if self.epoch < 1 or self.optimizer_step < 0 or self.samples_seen < 1:
            raise ValueError("invalid evaluation counters")
        if self.duration_s <= 0 or self.train.split != "train":
            raise ValueError("invalid evaluation duration or train population")
        if self.checkpoint.validation != self.validation:
            raise ValueError("evaluation and checkpoint validation disagree")
        if self.checkpoint.path != "checkpoints/last.pt":
            raise ValueError("evaluation must identify the current last checkpoint")
        identity = self.checkpoint.identity
        if (identity.epoch, identity.step) != (self.epoch, self.optimizer_step):
            raise ValueError("evaluation and checkpoint counters disagree")
        if (
            self.train.checkpoint_hash != identity.sha256
            or self.train.dataset_hash != identity.training_dataset_hash
            or self.train.task != identity.kind
        ):
            raise ValueError("train evaluation checkpoint identity disagrees")
        return self


type HistoryRecord = StepRecord | EpochRecord | EvaluationRecord


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class TrainingExecution:
    """Execution metadata, not a quality summary; checkpoints retain their own scores."""

    kind: Task
    arch: str
    dataset_hash: str
    conditioning: str
    config: dict[str, object]
    provenance: dict[str, object]
    metric: str
    status: RunStatus = "RUNNING"
    epoch: StrictInt = 0
    optimizer_step: StrictInt = 0
    samples_seen: StrictInt = 0
    stop_reason: str | None = None
    best: CheckpointRecord | None = None
    last: CheckpointRecord | None = None
    selected: CheckpointRecord | None = None
    selection: Literal["best", "last"] = "best"
    amp_enabled: StrictBool = False
    warnings: tuple[str, ...] = field(default=())
    schema_version: Literal[1] = 1
    execution_kind: Literal["TRAINING_EXECUTION"] = "TRAINING_EXECUTION"

    @model_validator(mode="after")
    def _bounds(self) -> TrainingExecution:
        if not is_sha256(self.dataset_hash) or not self.arch.strip() or not self.conditioning:
            raise ValueError("invalid training execution identity")
        if min(self.epoch, self.optimizer_step, self.samples_seen) < 0:
            raise ValueError("execution counters must be nonnegative")
        for checkpoint in (self.best, self.last, self.selected):
            if checkpoint is not None and (
                checkpoint.identity.kind != self.kind
                or checkpoint.identity.arch != self.arch
                or checkpoint.identity.training_dataset_hash != self.dataset_hash
                or checkpoint.metric != self.metric
                or checkpoint.identity.epoch is None
                or checkpoint.identity.epoch > self.epoch
                or checkpoint.identity.step is None
                or checkpoint.identity.step > self.optimizer_step
            ):
                raise ValueError("execution and checkpoint identity disagree")
        expected = self.best if self.selection == "best" else self.last
        if self.selected is not None and self.selected != expected:
            raise ValueError("selected checkpoint disagrees with selection policy")
        if self.status == "COMPLETED" and (
            self.best is None or self.last is None or self.selected is None
        ):
            raise ValueError("completed training requires best, last and selected checkpoints")
        if self.status != "RUNNING" and not self.stop_reason:
            raise ValueError("terminal execution requires a stop reason")
        return self


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Parsed captured records; incomplete execution is not relabeled as complete."""

    execution: TrainingExecution
    records: tuple[HistoryRecord, ...]
    warnings: tuple[str, ...]


def _parse_record(raw: bytes) -> HistoryRecord:
    """Decode a tagged strict history record without coercing unknown event kinds."""
    import json

    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("history event must be an object")
    if payload.get("record_kind") == "STEP":
        return TypeAdapter(StepRecord).validate_python(payload)
    if payload.get("record_kind") == "EPOCH":
        return TypeAdapter(EpochRecord).validate_python(payload)
    if payload.get("record_kind") == "EVALUATION":
        return TypeAdapter(EvaluationRecord).validate_python(payload)
    raise ValueError("unknown history record_kind")


def read_training_history(run: Path) -> Result[TrainingHistory, str]:
    """Read frozen history without model/dataset access, retaining incomplete prefixes."""
    try:
        execution = TypeAdapter(TrainingExecution).validate_json(
            (run / "execution.json").read_bytes()
        )
        path = run / "history.jsonl"
        raw = path.read_bytes() if path.exists() else b""
        lines = raw.splitlines(keepends=True)
        warnings: list[str] = []
        if lines and not lines[-1].endswith(b"\n"):
            if execution.status == "COMPLETED":
                return Err("completed history has an unterminated final record")
            lines.pop()
            warnings.append("Ignored an unterminated final record from an incomplete execution.")
        records: list[HistoryRecord] = []
        steps: list[StepRecord] = []
        last_epoch = 0
        last_batch = 0
        last_step = 0
        last_samples = 0
        reduced_epochs: set[int] = set()
        evaluated_epochs: set[int] = set()
        best: CheckpointRecord | None = None
        last: CheckpointRecord | None = None
        for line in lines:
            record = _parse_record(line)
            if record.epoch < last_epoch or record.optimizer_step < last_step:
                return Err("history counters are not monotonic")
            if isinstance(record, StepRecord):
                if record.epoch in reduced_epochs:
                    return Err("step follows an epoch reduction")
                if record.epoch != last_epoch:
                    steps = []
                if record.batch != last_batch + 1:
                    return Err("history batch counter is not contiguous")
                if record.optimizer_step != last_step + int(record.update_applied):
                    return Err("optimizer step disagrees with applied updates")
                if record.samples_seen != last_samples + record.n_samples:
                    return Err("history sample counter is not contiguous")
                steps.append(record)
                last_batch = record.batch
                last_step = record.optimizer_step
                last_samples = record.samples_seen
            elif isinstance(record, EpochRecord):
                reduced = reduce_epoch(tuple(steps))
                if record.epoch in reduced_epochs or isinstance(reduced, Err):
                    return Err("duplicate or unsupported epoch reduction")
                if reduced.value != record:
                    return Err("epoch evidence disagrees with captured step records")
                reduced_epochs.add(record.epoch)
            else:
                if record.epoch not in reduced_epochs or record.epoch in evaluated_epochs:
                    return Err("evaluation lacks a unique captured epoch")
                if record.samples_seen != last_samples or record.optimizer_step != last_step:
                    return Err("evaluation counters disagree with captured steps")
                checkpoint = record.checkpoint
                if checkpoint.metric != execution.metric:
                    return Err("history and execution selection metrics disagree")
                expected = is_improvement(
                    checkpoint.value,
                    best.value if best is not None else None,
                    checkpoint.direction,
                )
                if record.improved != expected:
                    return Err("history improvement flag disagrees with validation scores")
                if expected:
                    best = replace(checkpoint, path="checkpoints/best.pt")
                last = checkpoint
                evaluated_epochs.add(record.epoch)
            last_epoch = record.epoch
            records.append(record)
        if execution.status == "COMPLETED" and (
            not records
            or not isinstance(records[-1], EvaluationRecord)
            or (last_epoch, last_step, last_samples)
            != (execution.epoch, execution.optimizer_step, execution.samples_seen)
            or execution.last != records[-1].checkpoint
            or execution.best != best
            or execution.last != last
        ):
            return Err("completed execution disagrees with durable history")
        return Ok(TrainingHistory(execution, tuple(records), tuple(warnings)))
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        return Err(f"cannot read training history: {exc}")
