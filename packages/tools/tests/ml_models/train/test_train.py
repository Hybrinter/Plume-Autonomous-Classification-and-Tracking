"""Tests for the imperative evidence-recording training boundary."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import tools.ml_models.arch.registry as registry_mod
import tools.ml_models.train.loop as loop_mod
import torch
from flight.libs.types import Err, Ok, Result
from pydantic import TypeAdapter
from tools.ml_models.analysis import evaluate as evaluate_mod
from tools.ml_models.analysis.capture import CaptureSink
from tools.ml_models.analysis.config import EvaluationConfig
from tools.ml_models.analysis.contracts import (
    MetricSupport,
    MetricValue,
    SplitEvidence,
)
from tools.ml_models.analysis.training import (
    EvaluationRecord,
    StepRecord,
    TrainingExecution,
    read_training_history,
)
from tools.ml_models.dataset.augment import AugmentRecipe
from tools.ml_models.dataset.build import build_dataset
from tools.ml_models.dataset.loader import ShardDataset
from tools.ml_models.dataset.manifest import DatasetManifest
from tools.ml_models.dataset.raw import BinSpec, GsdPair, RawTile, RawTileRef
from tools.ml_models.dataset.spec import BuildSpec
from tools.ml_models.train.config import TrainConfig
from tools.ml_models.train.loop import train
from tools.ml_models.train.losses import LossComponents, PlumeLoss

_TILE_HW = (8, 10)
_BANDS: tuple[str, ...] = ("pan", "red", "nir")
_GSD_WINDOW = (
    GsdPair(15.0, 15.0),
    GsdPair(18.0, 21.0),
    GsdPair(25.0, 30.0),
)
_CHECKPOINT_KEYS = {
    "kind",
    "arch",
    "state_dict",
    "epoch",
    "conditioning",
    "config",
    "provenance",
    "dataset_hash",
    "step",
    "samples_seen",
    "metric",
    "direction",
    "value",
    "validation",
}


class _TinySource:
    """Planted-blob ``RawSource`` on tiny 8x10 tiles; every group has both classes."""

    name = "tiny"
    band_names: tuple[str, ...] = _BANDS
    domain = "unit"
    source_ref = ""
    bins: tuple[BinSpec, ...] = ()

    def __init__(self, n: int = 12, seed: int = 0) -> None:
        if n < 3:
            raise ValueError(f"fixture source needs at least 3 tiles; got {n}")
        self._tiles = _generate(n, seed)

    def index(self) -> tuple[RawTileRef, ...]:
        return tuple(tile.ref for tile in self._tiles)

    def iter_tiles(self) -> Iterator[RawTile]:
        yield from self._tiles


def _generate(n: int, seed: int) -> tuple[RawTile, ...]:
    height, width = _TILE_HW
    generator = np.random.default_rng(seed)
    tiles: list[RawTile] = []
    for index in range(n):
        label = float(index % 2)
        group = f"g{index // 3}"
        image = generator.random((3, height, width), dtype=np.float32) * np.float32(0.2)
        mask = np.zeros((1, height, width), dtype=np.uint8)
        if label:
            image[:, 2:6, 3:7] = np.float32(0.9)
            mask[0, 2:6, 3:7] = np.uint8(1)
        tiles.append(
            RawTile(
                ref=RawTileRef(
                    tile_id=f"t{index:03d}",
                    group_id=group,
                    label=label,
                    has_mask=True,
                    gsd=_GSD_WINDOW[index % len(_GSD_WINDOW)],
                    height=height,
                    width=width,
                    frame_id=group,
                    grid_rc=(index // 5, index % 5),
                    bin_id="",
                ),
                image=image,
                mask=mask,
            )
        )
    return tuple(tiles)


@pytest.fixture
def tiny_dataset() -> Callable[[Path], Path]:
    """Write an 8x10 planted-blob dataset; groups mix positive and empty rows."""

    def _build(dest: Path) -> Path:
        build_dataset(
            _TinySource(),
            dest,
            BuildSpec(augment=AugmentRecipe(elements=("id",))),
        )
        return dest

    return _build


def _cfg(dataset: Path, run_dir: Path, **overrides: object) -> TrainConfig:
    fields = {
        "dataset": str(dataset),
        "run_dir": str(run_dir),
        "device": "cpu",
        "epochs": 2,
        "batch_size": 2,
        "run_id": "run",
    }
    fields.update(overrides)
    return TrainConfig(**cast(Any, fields))


def _load_checkpoint(path: Path) -> dict[str, object]:
    return cast(dict[str, object], torch.load(path, map_location="cpu", weights_only=True))


def _execution(run: Path) -> TrainingExecution:
    return TypeAdapter(TrainingExecution).validate_json((run / "execution.json").read_bytes())


def _fake_evaluator(
    series: tuple[float, ...], metric: str
) -> tuple[Callable[..., Result[SplitEvidence, str]], list[str]]:
    """Stub ``evaluate_split``; ``series`` supplies successive val scores."""
    values = iter(series)
    splits: list[str] = []

    def fake(
        model: torch.nn.Module,
        dataset: Path,
        manifest: DatasetManifest,
        cfg: EvaluationConfig,
        objective: PlumeLoss | None = None,
        capture: CaptureSink | None = None,
    ) -> Result[SplitEvidence, str]:
        splits.append(cfg.split)
        score = next(values) if cfg.split == "val" else 0.0
        return Ok(
            SplitEvidence(
                task=cfg.kind,
                split=cfg.split,
                dataset_hash=manifest.dataset_hash,
                metrics=(
                    MetricValue(
                        name=metric,
                        value=score,
                        status="AVAILABLE",
                        support=MetricSupport(unit="IMAGE", n=1),
                    ),
                ),
            )
        )

    return fake, splits


def test_train_classifier_run(tmp_path: Path, tiny_dataset: Callable[[Path], Path]) -> None:
    """A tiny classifier run completes with durable records and both checkpoints."""
    dataset = tiny_dataset(tmp_path / "ds")
    run_dir = tmp_path / "runs"
    cfg = _cfg(dataset, run_dir, kind="classifier", arch="pactnet_w8_d2")
    result = train(cfg)
    assert isinstance(result, Ok), result
    run = result.value
    assert run == run_dir / "run"
    for name in (
        "checkpoints/last.pt",
        "checkpoints/best.pt",
        "config.toml",
        "execution.json",
        "history.jsonl",
        "resources.json",
    ):
        assert (run / name).is_file(), name
    assert not (run / "summary.json").exists()

    history = read_training_history(run)
    assert isinstance(history, Ok), history
    execution = history.value.execution
    assert execution.status == "COMPLETED"
    assert execution.stop_reason == "epochs"
    assert execution.metric == "average_precision"
    assert execution.best is not None and execution.last is not None
    assert execution.selected == execution.best
    records = history.value.records
    assert any(isinstance(record, StepRecord) for record in records)
    assert isinstance(records[-1], EvaluationRecord)
    assert records[-1].checkpoint == execution.last

    payload = _load_checkpoint(run / "checkpoints" / "last.pt")
    assert _CHECKPOINT_KEYS <= set(payload)
    assert payload["epoch"] == 2
    assert payload["metric"] == "average_precision"
    assert payload["direction"] == "MAXIMIZE"

    resources = json.loads((run / "resources.json").read_text())
    assert resources["records"], resources
    assert execution.last is not None
    assert resources["checkpoint"]["sha256"] == execution.last.identity.sha256
    assert resources["model_state"] == "last_evaluated_checkpoint"


def test_train_segmentor_run(tmp_path: Path, tiny_dataset: Callable[[Path], Path]) -> None:
    """A tiny segmentor run completes with the default positive-image IoU metric."""
    dataset = tiny_dataset(tmp_path / "ds")
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="segmentor",
        arch="dilatenet_w8_d2",
        epochs=1,
    )
    result = train(cfg)
    assert isinstance(result, Ok), result
    history = read_training_history(result.value)
    assert isinstance(history, Ok), history
    assert history.value.execution.status == "COMPLETED"
    assert history.value.execution.metric == "foreground_iou_mean_positive_images"


def test_training_never_touches_test_split(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shard tensors come only from train/val; evaluation only sees those splits."""
    dataset = tiny_dataset(tmp_path / "ds")
    shard_dirs: list[str] = []
    splits: list[str] = []
    original_init = ShardDataset.__init__
    original_evaluate = evaluate_mod.evaluate_split

    def spy_init(
        self: ShardDataset,
        shard_dir: str | Path,
        gsd_reference_m: float,
        task: str,
        *,
        channels: int,
    ) -> None:
        shard_dirs.append(str(shard_dir).replace("\\", "/"))
        original_init(self, shard_dir, gsd_reference_m, task, channels=channels)

    def spy_evaluate(
        model: torch.nn.Module,
        dataset_path: Path,
        manifest: DatasetManifest,
        cfg: EvaluationConfig,
        objective: PlumeLoss | None = None,
        capture: CaptureSink | None = None,
    ) -> Result[SplitEvidence, str]:
        splits.append(cfg.split)
        return original_evaluate(
            model, dataset_path, manifest, cfg, objective=objective, capture=capture
        )

    monkeypatch.setattr(ShardDataset, "__init__", spy_init)
    monkeypatch.setattr(evaluate_mod, "evaluate_split", spy_evaluate)

    cfg = _cfg(dataset, tmp_path / "runs", kind="classifier", arch="pactnet_w8_d2", epochs=1)
    assert isinstance(train(cfg), Ok)
    assert shard_dirs
    assert not any("/test/" in path for path in shard_dirs)
    assert set(splits) <= {"train", "val"}
    assert "val" in splits


def test_best_last_and_copy_with_minimize(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worsening val BCE keeps epoch 1 as best; the copy destination gets best.pt."""
    dataset = tiny_dataset(tmp_path / "ds")
    destination = tmp_path / "copied.pt"
    fake, _ = _fake_evaluator((0.2, 0.4, 0.6), "binary_cross_entropy")
    monkeypatch.setattr(evaluate_mod, "evaluate_split", fake)
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        epochs=3,
        eval_interval=1,
        val_metric="bce",
        checkpoint_path=str(destination),
    )
    result = train(cfg)
    assert isinstance(result, Ok), result
    history = read_training_history(result.value)
    assert isinstance(history, Ok), history
    execution = history.value.execution
    assert execution.status == "COMPLETED"
    assert execution.metric == "binary_cross_entropy"
    best = execution.best
    last = execution.last
    assert best is not None and last is not None
    assert best.identity.epoch == 1
    assert last.identity.epoch == 3
    assert best.identity.sha256 != last.identity.sha256
    assert best.value == pytest.approx(0.2)
    assert last.value == pytest.approx(0.6)
    assert execution.selected == best
    assert destination.read_bytes() == (result.value / "checkpoints" / "best.pt").read_bytes()


def test_best_last_with_maximize(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Maximize selects the highest val AP and never relabels later checkpoints."""
    dataset = tiny_dataset(tmp_path / "ds")
    fake, _ = _fake_evaluator((0.8, 0.6, 0.7), "average_precision")
    monkeypatch.setattr(evaluate_mod, "evaluate_split", fake)
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        epochs=3,
        eval_interval=1,
        val_metric="average_precision",
    )
    result = train(cfg)
    assert isinstance(result, Ok), result
    history = read_training_history(result.value)
    assert isinstance(history, Ok), history
    execution = history.value.execution
    best = execution.best
    last = execution.last
    assert best is not None and last is not None
    assert best.identity.epoch == 1
    assert best.value == pytest.approx(0.8)
    assert last.identity.epoch == 3
    assert last.value == pytest.approx(0.7)


def test_early_stopping_on_tie(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """patience=1 ends the run at the first non-improving evaluation."""
    dataset = tiny_dataset(tmp_path / "ds")
    fake, splits = _fake_evaluator((0.5, 0.5, 0.1), "binary_cross_entropy")
    monkeypatch.setattr(evaluate_mod, "evaluate_split", fake)
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        epochs=3,
        eval_interval=1,
        patience=1,
        val_metric="bce",
    )
    result = train(cfg)
    assert isinstance(result, Ok), result
    history = read_training_history(result.value)
    assert isinstance(history, Ok), history
    execution = history.value.execution
    assert execution.status == "COMPLETED"
    assert execution.stop_reason == "early_stopping"
    assert execution.epoch == 2
    assert splits.count("val") == 2


def test_max_steps_truncates_epoch_and_still_evaluates(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_steps=1 stops inside epoch 1 and forces a terminal evaluation."""
    dataset = tiny_dataset(tmp_path / "ds")
    fake, _ = _fake_evaluator((0.4,), "binary_cross_entropy")
    monkeypatch.setattr(evaluate_mod, "evaluate_split", fake)
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        epochs=3,
        eval_interval=5,
        max_steps=1,
        val_metric="bce",
    )
    result = train(cfg)
    assert isinstance(result, Ok), result
    history = read_training_history(result.value)
    assert isinstance(history, Ok), history
    execution = history.value.execution
    assert execution.status == "COMPLETED"
    assert execution.stop_reason == "max_steps"
    assert execution.optimizer_step == 1
    assert execution.epoch == 1
    records = history.value.records
    assert isinstance(records[-1], EvaluationRecord)


def test_last_saved_only_at_evaluated_epochs(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonevaluated epoch leaves no last.pt and no evaluation record."""
    dataset = tiny_dataset(tmp_path / "ds")
    calls: list[int] = []

    real_save = torch.save

    def counting_save(payload: object, target: object) -> None:
        calls.append(1)
        real_save(payload, cast(Any, target))

    monkeypatch.setattr(torch, "save", counting_save)
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        epochs=2,
        eval_interval=2,
    )
    result = train(cfg)
    assert isinstance(result, Ok), result
    history = read_training_history(result.value)
    assert isinstance(history, Ok), history
    executions = history.value.records
    evaluations = [r for r in executions if isinstance(r, EvaluationRecord)]
    assert len(evaluations) == 1
    assert evaluations[0].epoch == 2
    assert len(calls) == 1
    execution = history.value.execution
    assert execution.last is not None
    assert execution.last.identity.epoch == 2


def test_run_dir_refused_even_with_overwrite(
    tmp_path: Path, tiny_dataset: Callable[[Path], Path]
) -> None:
    """An existing run directory is never overwritten, even with overwrite=True."""
    dataset = tiny_dataset(tmp_path / "ds")
    existing = tmp_path / "runs" / "run"
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("untouched")
    for overwrite in (False, True):
        cfg = _cfg(
            dataset,
            tmp_path / "runs",
            kind="classifier",
            arch="pactnet_w8_d2",
            overwrite=overwrite,
        )
        result = train(cfg)
        assert isinstance(result, Err)
    assert marker.read_text() == "untouched"


def test_destinations_inside_dataset_refused(
    tmp_path: Path, tiny_dataset: Callable[[Path], Path]
) -> None:
    """Run and copy destinations under the dataset root fail before outputs."""
    dataset = tiny_dataset(tmp_path / "ds")
    inside = _cfg(
        dataset,
        dataset / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
    )
    assert isinstance(train(inside), Err)
    copy_inside = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        checkpoint_path=str(dataset / "stolen.pt"),
    )
    assert isinstance(train(copy_inside), Err)
    assert not (dataset / "runs").exists()


def test_run_id_and_destination_rejected(
    tmp_path: Path, tiny_dataset: Callable[[Path], Path]
) -> None:
    """Path components and existing copy destinations fail fast."""
    dataset = tiny_dataset(tmp_path / "ds")
    for bad in ("a/b", "..", "a:b", ".", "a\\b"):
        cfg = _cfg(dataset, tmp_path / "runs", run_id=bad)
        assert isinstance(train(cfg), Err), bad
    destination = tmp_path / "exists.pt"
    destination.write_bytes(b"claimed")
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        checkpoint_path=str(destination),
    )
    result = train(cfg)
    assert isinstance(result, Err)
    assert destination.read_bytes() == b"claimed"


def test_missing_and_blank_dataset_fail(
    tmp_path: Path, tiny_dataset: Callable[[Path], Path]
) -> None:
    """A missing dataset fails; a blank one fails before creating outputs."""
    assert isinstance(train(), Err)
    assert isinstance(train(TrainConfig(dataset="   ")), Err)
    missing = _cfg(tmp_path / "nope", tmp_path / "runs")
    assert isinstance(train(missing), Err)
    assert not (tmp_path / "runs").exists()


def test_nonfinite_loss_fails_before_optimizer_step(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NaN objective aborts without ever calling optimizer.step."""
    dataset = tiny_dataset(tmp_path / "ds")
    steps: list[int] = []
    original_step = torch.optim.SGD.step

    def counting_step(self: torch.optim.SGD, closure: object = None) -> object:
        steps.append(1)
        return original_step(self, closure)

    def nan_components(
        self: PlumeLoss, logits: torch.Tensor, targets: torch.Tensor
    ) -> LossComponents:
        del self, targets
        total = torch.full(
            (logits.shape[0],), float("nan"), dtype=logits.dtype, device=logits.device
        )
        return LossComponents(total=total, bce=None, focal=None, dice=None)

    monkeypatch.setattr(torch.optim.SGD, "step", counting_step)
    monkeypatch.setattr(PlumeLoss, "per_sample_components", nan_components)
    cfg = _cfg(dataset, tmp_path / "runs", kind="classifier", arch="pactnet_w8_d2")
    result = train(cfg)
    assert isinstance(result, Err)
    assert steps == []
    run = tmp_path / "runs" / "run"
    execution = json.loads((run / "execution.json").read_text())
    assert execution["status"] == "FAILED"


def test_keyboard_interrupt_marks_interrupted(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-run KeyboardInterrupt leaves INTERRUPTED status and prior records."""
    dataset = tiny_dataset(tmp_path / "ds")
    calls: list[int] = []
    original = PlumeLoss.per_sample_components

    def interrupting(
        self: PlumeLoss, logits: torch.Tensor, targets: torch.Tensor
    ) -> LossComponents:
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original(self, logits, targets)

    monkeypatch.setattr(PlumeLoss, "per_sample_components", interrupting)
    cfg = _cfg(dataset, tmp_path / "runs", kind="classifier", arch="pactnet_w8_d2")
    result = train(cfg)
    assert isinstance(result, Err)
    run = tmp_path / "runs" / "run"
    execution = json.loads((run / "execution.json").read_text())
    assert execution["status"] == "INTERRUPTED"
    history = read_training_history(run)
    assert isinstance(history, Ok), history
    assert history.value.records
    assert all(isinstance(record, StepRecord) for record in history.value.records)


def test_runtime_error_marks_failed(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-run RuntimeError leaves FAILED status and retained history."""
    dataset = tiny_dataset(tmp_path / "ds")
    calls: list[int] = []
    original = PlumeLoss.per_sample_components

    def exploding(self: PlumeLoss, logits: torch.Tensor, targets: torch.Tensor) -> LossComponents:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected failure")
        return original(self, logits, targets)

    monkeypatch.setattr(PlumeLoss, "per_sample_components", exploding)
    cfg = _cfg(dataset, tmp_path / "runs", kind="classifier", arch="pactnet_w8_d2")
    result = train(cfg)
    assert isinstance(result, Err)
    run = tmp_path / "runs" / "run"
    execution = json.loads((run / "execution.json").read_text())
    assert execution["status"] == "FAILED"
    history = read_training_history(run)
    assert isinstance(history, Ok), history
    assert history.value.records


def test_seed_repeats_deterministically(
    tmp_path: Path, tiny_dataset: Callable[[Path], Path]
) -> None:
    """Two identical configs produce identical parameters and non-timing records."""
    dataset = tiny_dataset(tmp_path / "ds")
    runs: list[Path] = []
    for run_id in ("first", "second"):
        cfg = _cfg(
            dataset,
            tmp_path / "runs",
            kind="classifier",
            arch="pactnet_w8_d2",
            epochs=1,
            run_id=run_id,
            seed=7,
        )
        result = train(cfg)
        assert isinstance(result, Ok), result
        runs.append(result.value)
    first = _load_checkpoint(runs[0] / "checkpoints" / "last.pt")
    second = _load_checkpoint(runs[1] / "checkpoints" / "last.pt")
    first_state = first["state_dict"]
    second_state = second["state_dict"]
    assert isinstance(first_state, dict) and isinstance(second_state, dict)
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        assert torch.equal(first_state[name], second_state[name]), name
    histories = []
    for run in runs:
        history = read_training_history(run)
        assert isinstance(history, Ok)
        histories.append(history.value.records)
    assert len(histories[0]) == len(histories[1])
    for left, right in zip(*histories, strict=True):
        assert type(left) is type(right)
        if isinstance(left, StepRecord):
            assert isinstance(right, StepRecord)
            assert left.loss == right.loss
            assert left.samples_seen == right.samples_seen
            assert left.learning_rates == right.learning_rates


def test_amp_on_cpu_records_warning_not_fake_scales(
    tmp_path: Path, tiny_dataset: Callable[[Path], Path]
) -> None:
    """amp=True on CPU warns, disables AMP, and logs no invented scales."""
    dataset = tiny_dataset(tmp_path / "ds")
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        epochs=1,
        amp=True,
        amp_diagnostics=True,
    )
    result = train(cfg)
    assert isinstance(result, Ok), result
    history = read_training_history(result.value)
    assert isinstance(history, Ok), history
    execution = history.value.execution
    assert execution.amp_enabled is False
    assert any("amp" in warning.lower() for warning in execution.warnings)
    steps = [r for r in history.value.records if isinstance(r, StepRecord)]
    assert steps
    assert all(step.amp_scale_before is None and step.amp_scale_after is None for step in steps)


def test_cli_train_rejects_plural_dataset_and_missing_config(tmp_path: Path) -> None:
    """A repeated ``--dataset`` or a missing config still fails fast."""
    from tools.ml_models.cli import main

    dataset = str(tmp_path / "ds")
    assert main(["train", "--dataset", dataset, "--dataset", dataset]) != 0
    assert main(["train", "--config", str(tmp_path / "missing.toml")]) != 0


def test_cli_train_success(tmp_path: Path, tiny_dataset: Callable[[Path], Path]) -> None:
    """``ml-models train`` runs a tiny job and echoes the run directory."""
    from tools.ml_models.cli import main

    dataset = tiny_dataset(tmp_path / "ds")
    code = main(
        [
            "train",
            "--kind",
            "classifier",
            "--arch",
            "pactnet_w8_d2",
            "--dataset",
            str(dataset),
            "--run-dir",
            str(tmp_path / "runs"),
            "--run-id",
            "cli-run",
            "--epochs",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    assert (tmp_path / "runs" / "cli-run" / "execution.json").is_file()


def test_unavailable_selected_metric_fails(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An UNAVAILABLE validation metric aborts without fabricated checkpoints."""
    dataset = tiny_dataset(tmp_path / "ds")

    def fake(
        model: torch.nn.Module,
        dataset: Path,
        manifest: DatasetManifest,
        cfg: EvaluationConfig,
        objective: PlumeLoss | None = None,
        capture: CaptureSink | None = None,
    ) -> Result[SplitEvidence, str]:
        metric = (
            MetricValue(
                name="average_precision",
                value=0.5,
                status="AVAILABLE",
                support=MetricSupport(unit="IMAGE", n=1),
            )
            if cfg.split == "train"
            else MetricValue(
                name="average_precision",
                value=None,
                status="UNAVAILABLE",
                reason="no positive validation examples",
                support=MetricSupport(unit="IMAGE", n=0),
            )
        )
        return Ok(
            SplitEvidence(
                task=cfg.kind,
                split=cfg.split,
                dataset_hash=manifest.dataset_hash,
                metrics=(metric,),
            )
        )

    monkeypatch.setattr(evaluate_mod, "evaluate_split", fake)
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        epochs=1,
    )
    result = train(cfg)
    assert isinstance(result, Err)
    execution = _execution(tmp_path / "runs" / "run")
    assert execution.status == "FAILED"
    assert execution.best is None and execution.last is None
    assert not (tmp_path / "runs" / "run" / "checkpoints" / "last.pt").exists()


def test_nonfinite_gradient_fails_before_optimizer_step(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NaN gradient aborts the batch before the optimizer update."""
    dataset = tiny_dataset(tmp_path / "ds")
    real_build = registry_mod.build
    steps: list[None] = []

    def poisoned(kind: str, arch: str, in_channels: int) -> torch.nn.Module:
        model = real_build(kind, arch, in_channels)
        for parameter in model.parameters():
            if parameter.requires_grad:
                cast(Any, parameter).register_hook(lambda grad: torch.full_like(grad, float("nan")))
                break
        return model

    def counting_step(self: torch.optim.SGD, *args: object, **kwargs: object) -> None:
        steps.append(None)

    monkeypatch.setattr(registry_mod, "build", poisoned)
    monkeypatch.setattr(torch.optim.SGD, "step", counting_step)
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        epochs=1,
    )
    result = train(cfg)
    assert isinstance(result, Err)
    assert "non-finite gradients" in result.error
    assert steps == []
    execution = _execution(tmp_path / "runs" / "run")
    assert execution.status == "FAILED"
    assert execution.optimizer_step == 0


def test_selected_checkpoint_last_copies_last(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """selected_checkpoint='last' publishes the last evaluated checkpoint."""
    dataset = tiny_dataset(tmp_path / "ds")
    destination = tmp_path / "copied.pt"
    fake, _ = _fake_evaluator((0.2, 0.4, 0.6), "binary_cross_entropy")
    monkeypatch.setattr(evaluate_mod, "evaluate_split", fake)
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        epochs=3,
        eval_interval=1,
        val_metric="bce",
        selected_checkpoint="last",
        checkpoint_path=str(destination),
    )
    result = train(cfg)
    assert isinstance(result, Ok), result
    history = read_training_history(result.value)
    assert isinstance(history, Ok), history
    execution = history.value.execution
    best = execution.best
    last = execution.last
    assert best is not None and last is not None
    assert execution.selected == last
    assert best.identity.epoch == 1 and last.identity.epoch == 3
    assert destination.read_bytes() == (result.value / "checkpoints" / "last.pt").read_bytes()


def test_gradient_diagnostics_records_finite_norms(
    tmp_path: Path, tiny_dataset: Callable[[Path], Path]
) -> None:
    """Gradient diagnostics record a finite nonnegative L2 norm per step."""
    dataset = tiny_dataset(tmp_path / "ds")
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        epochs=1,
        gradient_diagnostics=True,
    )
    result = train(cfg)
    assert isinstance(result, Ok), result
    history = read_training_history(result.value)
    assert isinstance(history, Ok), history
    steps = tuple(record for record in history.value.records if isinstance(record, StepRecord))
    assert steps
    for record in steps:
        assert record.gradient_norm is not None
        assert record.gradient_norm >= 0.0
        assert record.amp_scale_before is None
        assert record.amp_scale_after is None


def test_adamw_cosine_lr_trajectory(tmp_path: Path, tiny_dataset: Callable[[Path], Path]) -> None:
    """AdamW plus the cosine scheduler produce a decaying epoch-two LR."""
    dataset = tiny_dataset(tmp_path / "ds")
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        epochs=2,
        learning_rate=0.01,
        optimizer="adamw",
        scheduler="cosine",
    )
    result = train(cfg)
    assert isinstance(result, Ok), result
    history = read_training_history(result.value)
    assert isinstance(history, Ok), history
    steps = tuple(record for record in history.value.records if isinstance(record, StepRecord))
    first = [record.learning_rates[0] for record in steps if record.epoch == 1]
    second = [record.learning_rates[0] for record in steps if record.epoch == 2]
    assert first and second
    assert all(value == pytest.approx(0.01) for value in first)
    assert all(0.0 <= value < 0.01 for value in second)


def test_publish_preserves_user_tmp_and_is_independent(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication leaves preexisting sibling bytes and copies, not links, bytes."""
    dataset = tiny_dataset(tmp_path / "ds")
    destination = tmp_path / "copied.pt"
    user_tmp = tmp_path / "copied.pt.tmp"
    user_tmp.write_bytes(b"user-owned bytes")
    fake, _ = _fake_evaluator((0.5,), "binary_cross_entropy")
    monkeypatch.setattr(evaluate_mod, "evaluate_split", fake)
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        epochs=1,
        val_metric="bce",
        checkpoint_path=str(destination),
    )
    result = train(cfg)
    assert isinstance(result, Ok), result
    assert user_tmp.read_bytes() == b"user-owned bytes"
    selected = result.value / "checkpoints" / "best.pt"
    assert destination.read_bytes() == selected.read_bytes()
    assert not os.path.samefile(destination, selected)
    selected.write_bytes(b"mutated after publication")
    assert destination.read_bytes() != b"mutated after publication"


def test_publish_race_preserves_destination_and_user_tmp(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A destination created mid-publication refuses and stays untouched."""
    dataset = tiny_dataset(tmp_path / "ds")
    destination = tmp_path / "copied.pt"
    user_tmp = tmp_path / "copied.pt.tmp"
    user_tmp.write_bytes(b"user-owned bytes")
    fake, _ = _fake_evaluator((0.5,), "binary_cross_entropy")
    monkeypatch.setattr(evaluate_mod, "evaluate_split", fake)

    def racing_link(src: object, dst: object) -> None:
        Path(str(dst)).write_bytes(b"concurrent winner")
        raise FileExistsError("destination exists")

    monkeypatch.setattr(os, "link", racing_link)
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        epochs=1,
        val_metric="bce",
        checkpoint_path=str(destination),
    )
    result = train(cfg)
    assert isinstance(result, Err)
    assert destination.read_bytes() == b"concurrent winner"
    assert user_tmp.read_bytes() == b"user-owned bytes"
    execution = _execution(tmp_path / "runs" / "run")
    assert execution.status == "FAILED"


def test_setup_failure_marks_failed(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A setup failure after reservation persists a FAILED execution."""
    dataset = tiny_dataset(tmp_path / "ds")

    def broken(path: Path, cfg: TrainConfig) -> None:
        raise RuntimeError("config write exploded")

    monkeypatch.setattr(loop_mod, "write_train_config_toml", broken)
    cfg = _cfg(dataset, tmp_path / "runs", kind="classifier", arch="pactnet_w8_d2")
    result = train(cfg)
    assert isinstance(result, Err)
    run = tmp_path / "runs" / "run"
    execution = _execution(run)
    assert execution.status == "FAILED"
    assert "config write exploded" in (execution.stop_reason or "")
    assert not (run / "history.jsonl").exists()


def test_setup_interrupt_marks_interrupted(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A KeyboardInterrupt during setup persists an INTERRUPTED execution."""
    dataset = tiny_dataset(tmp_path / "ds")

    def broken(path: Path, cfg: TrainConfig) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(loop_mod, "write_train_config_toml", broken)
    cfg = _cfg(dataset, tmp_path / "runs", kind="classifier", arch="pactnet_w8_d2")
    result = train(cfg)
    assert isinstance(result, Err)
    execution = _execution(tmp_path / "runs" / "run")
    assert execution.status == "INTERRUPTED"


def test_failed_best_copy_keeps_truthful_records(
    tmp_path: Path,
    tiny_dataset: Callable[[Path], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed best copy keeps the earlier best and the actual last hash."""
    dataset = tiny_dataset(tmp_path / "ds")
    real_copy = loop_mod._copy_file
    calls: list[int] = []

    def flaky_copy(source: Path, destination: Path) -> None:
        calls.append(1)
        if len(calls) == 2:
            raise OSError("injected copy failure")
        real_copy(source, destination)

    fake, _ = _fake_evaluator((0.5, 0.4, 0.3), "binary_cross_entropy")
    monkeypatch.setattr(evaluate_mod, "evaluate_split", fake)
    monkeypatch.setattr(loop_mod, "_copy_file", flaky_copy)
    cfg = _cfg(
        dataset,
        tmp_path / "runs",
        kind="classifier",
        arch="pactnet_w8_d2",
        epochs=3,
        eval_interval=1,
        val_metric="bce",
    )
    result = train(cfg)
    assert isinstance(result, Err)
    run = tmp_path / "runs" / "run"
    execution = _execution(run)
    assert execution.status == "FAILED"
    best = execution.best
    last = execution.last
    assert best is not None and last is not None
    assert best.identity.epoch == 1
    assert last.identity.epoch == 2
    assert (
        best.identity.sha256
        == hashlib.sha256((run / "checkpoints" / "best.pt").read_bytes()).hexdigest()
    )
    assert (
        last.identity.sha256
        == hashlib.sha256((run / "checkpoints" / "last.pt").read_bytes()).hexdigest()
    )
    assert execution.selected == best
