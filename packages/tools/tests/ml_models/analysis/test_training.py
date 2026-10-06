"""Independent history-reduction and selection-policy examples."""

import json
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.contracts import (
    CheckpointIdentity,
    MetricSupport,
    MetricValue,
    SplitEvidence,
)
from tools.ml_models.analysis.training import (
    CheckpointRecord,
    EvaluationRecord,
    LossObservation,
    StepRecord,
    TrainingExecution,
    is_improvement,
    observe_loss,
    read_training_history,
    reduce_epoch,
    selected_metric,
)
from tools.ml_models.train.config import validation_metric
from tools.ml_models.train.losses import LossComponents

_HASH = "a" * 64


def _step(
    batch: int = 1,
    *,
    epoch: int = 1,
    n: int = 2,
    total: float = 3.0,
    bce: float = 1.0,
    dice: float = 2.0,
    step: int = 1,
    seen: int = 2,
    applied: bool = True,
    duration: float = 2.0,
) -> StepRecord:
    return StepRecord(
        epoch=epoch,
        batch=batch,
        optimizer_step=step,
        samples_seen=seen,
        n_samples=n,
        loss=LossObservation(total=total, bce=bce, dice=dice),
        learning_rates=(0.01,),
        duration_s=duration,
        throughput_images_s=n / duration,
        update_applied=applied,
    )


def _evidence(
    name: str = "average_precision",
    value: float | None = 0.8,
    *,
    split: str = "val",
    checkpoint_hash: str | None = None,
) -> SplitEvidence:
    from typing import cast

    from tools.ml_models.analysis.contracts import Split

    return SplitEvidence(
        task="classifier",
        split=cast(Split, split),
        dataset_hash=_HASH,
        checkpoint_hash=checkpoint_hash,
        support=MetricSupport(unit="IMAGE", n=2),
        metrics=(
            MetricValue(
                name=name,
                value=value,
                status="AVAILABLE" if value is not None else "UNAVAILABLE",
                reason=None if value is not None else "no positive examples",
                support=MetricSupport(unit="IMAGE", n=2),
            ),
        ),
    )


def _checkpoint(epoch: int = 1, step: int = 1, value: float = 0.8) -> CheckpointRecord:
    digest = str(epoch) * 64
    return CheckpointRecord(
        path="checkpoints/last.pt",
        identity=CheckpointIdentity(
            sha256=digest,
            kind="classifier",
            arch="tiny",
            training_dataset_hash=_HASH,
            epoch=epoch,
            step=step,
        ),
        metric="average_precision",
        direction="MAXIMIZE",
        value=value,
        validation=_evidence(value=value, checkpoint_hash=digest),
    )


def _execution(**changes: object) -> TrainingExecution:
    from pydantic import TypeAdapter

    data: dict[str, object] = {
        "kind": "classifier",
        "arch": "tiny",
        "dataset_hash": _HASH,
        "conditioning": "film-log-gsd-v1",
        "config": {},
        "provenance": {},
        "metric": "average_precision",
    }
    return TypeAdapter(TrainingExecution).validate_python(data | changes)


def _write(run: Path, execution: TrainingExecution, records: list[object], tail: str = "") -> None:
    (run / "execution.json").write_text(json.dumps(asdict(execution)), encoding="utf-8")
    (run / "history.jsonl").write_text(
        "".join(json.dumps(asdict(cast(Any, record))) + "\n" for record in records) + tail,
        encoding="utf-8",
    )


def test_observation_uses_weighted_terms_without_touching_gradients() -> None:
    bce = torch.tensor([1.0, 3.0], requires_grad=True)
    dice = torch.tensor([2.0, 4.0], requires_grad=True)
    components = LossComponents(total=2 * bce + 0.5 * dice, bce=bce, focal=None, dice=dice)
    measured = observe_loss(components, 2.0, 0.5)
    assert isinstance(measured, Ok)
    assert measured.value == LossObservation(total=5.5, bce=4.0, dice=1.5)
    assert bce.grad is None and dice.grad is None
    cast(Callable[[], None], components.total.mean().backward)()
    assert bce.grad is not None and torch.equal(bce.grad, torch.ones(2))
    assert dice.grad is not None and torch.equal(dice.grad, torch.full((2,), 0.25))


def test_observation_rejects_nonfinite_or_misaligned_terms() -> None:
    for vector in (torch.tensor([float("nan")]), torch.tensor([[1.0]])):
        result = observe_loss(LossComponents(vector, vector, None, None), 1.0, 1.0)
        assert isinstance(result, Err)


def test_unequal_batches_are_weighted_by_sample_count_and_skips_remain_exposure() -> None:
    first = _step()
    skipped = _step(
        2,
        n=1,
        total=9.0,
        bce=3.0,
        dice=6.0,
        step=1,
        seen=3,
        applied=False,
        duration=1.0,
    )
    reduced = reduce_epoch((first, skipped))
    assert isinstance(reduced, Ok)
    epoch = reduced.value
    assert epoch.loss.total == 5.0
    assert epoch.loss.bce == pytest.approx(5 / 3)
    assert epoch.loss.dice == pytest.approx(10 / 3)
    assert (epoch.n_samples, epoch.n_batches, epoch.n_updates) == (3, 2, 1)
    assert epoch.duration_s == 3.0 and epoch.throughput_images_s == 1.0
    assert epoch.optimizer_step == 1 and epoch.samples_seen == 3
    assert isinstance(reduce_epoch(()), Err)
    assert isinstance(reduce_epoch((first, replace(skipped, epoch=2))), Err)
    inactive = replace(skipped, loss=LossObservation(total=9.0, bce=9.0))
    assert isinstance(reduce_epoch((first, inactive)), Err)


def test_loss_records_reject_false_component_sums_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="sum"):
        LossObservation(total=2.0, bce=1.0, dice=2.0)
    with pytest.raises(ValueError):
        LossObservation(total=True, bce=1.0)
    with pytest.raises(ValueError):
        LossObservation(total=float("inf"), bce=float("inf"))
    with pytest.raises(ValueError, match="throughput"):
        replace(_step(), throughput_images_s=99.0)


def test_checkpoint_defaults_aliases_and_direction_are_explicit() -> None:
    assert validation_metric("classifier") == "average_precision"
    assert validation_metric("segmentor") == "foreground_iou_mean_positive_images"
    assert validation_metric("classifier", "pr_auc") == "average_precision"
    assert validation_metric("segmentor", "mean_iou") == "foreground_iou_mean_all_annotated_images"
    for name, value, direction in (
        ("average_precision", 0.8, "MAXIMIZE"),
        ("binary_cross_entropy", 0.4, "MINIMIZE"),
        ("brier_score", 0.1, "MINIMIZE"),
    ):
        result = selected_metric(_evidence(name, value), name)
        assert isinstance(result, Ok)
        assert result.value == (value, direction)
    assert is_improvement(0.5, None, "MAXIMIZE")
    assert is_improvement(0.5, None, "MINIMIZE")
    assert not is_improvement(0.5, 0.5, "MAXIMIZE")
    assert is_improvement(0.4, 0.5, "MINIMIZE")
    assert not is_improvement(0.4, 0.5, "MAXIMIZE")


def test_unavailable_absent_descriptive_or_nonvalidation_metric_cannot_select() -> None:
    assert isinstance(selected_metric(_evidence(value=None), "average_precision"), Err)
    assert isinstance(selected_metric(_evidence(), "binary_cross_entropy"), Err)
    assert isinstance(selected_metric(_evidence(split="test"), "average_precision"), Err)
    assert isinstance(
        selected_metric(
            _evidence("predicted_positive_fraction", 0.5), "predicted_positive_fraction"
        ),
        Err,
    )


def test_checkpoint_identity_and_validation_must_agree() -> None:
    checkpoint = _checkpoint()
    with pytest.raises(ValueError, match="score"):
        replace(checkpoint, value=0.1)
    with pytest.raises(ValueError, match="hash"):
        replace(checkpoint, validation=replace(checkpoint.validation, checkpoint_hash="f" * 64))
    with pytest.raises(ValueError, match="dataset"):
        replace(checkpoint, validation=replace(checkpoint.validation, dataset_hash="b" * 64))
    with pytest.raises(ValueError, match="completed training"):
        _execution(status="COMPLETED", stop_reason="epochs")


def test_incomplete_reader_retains_complete_prefix_but_never_accepts_corrupt_lines(
    tmp_path: Path,
) -> None:
    _write(tmp_path, _execution(status="INTERRUPTED", stop_reason="interrupt"), [_step()], '{"')
    parsed = read_training_history(tmp_path)
    assert isinstance(parsed, Ok)
    assert parsed.value.execution.status == "INTERRUPTED"
    assert parsed.value.records == (_step(),)
    assert parsed.value.warnings
    _write(tmp_path, _execution(status="FAILED", stop_reason="io"), [_step()], '{"}\n')
    assert isinstance(read_training_history(tmp_path), Err)


def test_completed_reader_proves_best_vs_last_and_improvement_flags(tmp_path: Path) -> None:
    records: list[object] = []
    checkpoints: list[CheckpointRecord] = []
    for epoch, value in ((1, 0.8), (2, 0.6)):
        step = _step(epoch, epoch=epoch, step=epoch, seen=epoch * 2)
        reduced = reduce_epoch((step,))
        assert isinstance(reduced, Ok)
        checkpoint = _checkpoint(epoch, epoch, value)
        checkpoints.append(checkpoint)
        evaluated = EvaluationRecord(
            epoch=epoch,
            optimizer_step=epoch,
            samples_seen=epoch * 2,
            train=_evidence(split="train", checkpoint_hash=checkpoint.identity.sha256),
            validation=checkpoint.validation,
            checkpoint=checkpoint,
            improved=epoch == 1,
            duration_s=1.0,
        )
        records.extend([step, reduced.value, evaluated])
    best = replace(checkpoints[0], path="checkpoints/best.pt")
    execution = _execution(
        status="COMPLETED",
        stop_reason="epochs",
        epoch=2,
        optimizer_step=2,
        samples_seen=4,
        best=best,
        last=checkpoints[-1],
        selected=best,
    )
    _write(tmp_path, execution, records)
    parsed = read_training_history(tmp_path)
    assert isinstance(parsed, Ok)
    assert parsed.value.execution.best == best
    bad_final = records[-1]
    assert isinstance(bad_final, EvaluationRecord)
    _write(tmp_path, execution, records[:-1] + [replace(bad_final, improved=True)])
    assert isinstance(read_training_history(tmp_path), Err)
    _write(tmp_path, execution, records, "{}")
    assert isinstance(read_training_history(tmp_path), Err)
    _write(tmp_path, execution, records[:-1])
    assert isinstance(read_training_history(tmp_path), Err)


def test_reader_rejects_wrong_epoch_reductions_or_update_counts(tmp_path: Path) -> None:
    step = _step()
    epoch = reduce_epoch((step,))
    assert isinstance(epoch, Ok)
    _write(
        tmp_path, _execution(), [step, replace(epoch.value, loss=LossObservation(total=4, bce=4))]
    )
    assert isinstance(read_training_history(tmp_path), Err)
    _write(tmp_path, _execution(), [replace(step, optimizer_step=2)])
    assert isinstance(read_training_history(tmp_path), Err)
