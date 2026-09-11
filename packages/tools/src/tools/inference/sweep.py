"""Cartesian hyperparameter sweep over the local run catalog.

A space TOML uses TrainConfig field names. Scalar keys are fixed. List keys
are search axes. Trials expand in sorted-axis cartesian order and stop at
``max_runs`` when that key is set and positive. Each trial trains, scores the
val split, and appends one JSONL record.

A sweep large enough to be worth running is also large enough that something in
it will fail: one architecture will exhaust VRAM at a batch size the others
tolerate, or the machine will be interrupted partway through. Two behaviours
keep such a sweep useful rather than lost. A failed trial records its error and
the sweep moves to the next one, and ``resume`` reads the existing JSONL and
skips the trials it already holds.

A third behaviour guards the output itself. A sweep holds a lock file beside its
JSONL, so a second sweep aimed at the same output refuses to start rather than
interleaving rows and training into the same run directories.

Contains:
  - load_sweep_space: expand a space file into TrainConfig trials.
  - completed_run_ids: run identifiers already present in a JSONL.
  - sweep: run trials and write sweep.jsonl.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
import os
import time
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import torch

from tools.inference.arch.registry import resolve_arch
from tools.inference.eval import evaluate
from tools.inference.train import TrainConfig, apply_train_mapping, config_digest, train

_SPACE_EXTRA = frozenset({"max_runs"})

# Recorded verbatim from each trial config so the JSONL alone is enough to
# rank runs and reproduce one, without reopening every run directory.
_TRIAL_FIELDS: tuple[str, ...] = (
    "kind",
    "arch",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "optimizer",
    "scheduler",
    "loss",
    "pos_weight",
    "augment",
    "shuffle",
    "amp",
    "seed",
)


def load_sweep_space(path: str | Path) -> tuple[TrainConfig, ...]:
    """Expand a space TOML into a tuple of TrainConfig trials.

    Args:
        path: TOML file. Keys are TrainConfig fields plus optional ``max_runs``.

    Returns:
        tuple[TrainConfig, ...]: Cartesian product, truncated to ``max_runs``.

    Raises:
        ValueError: If a key is unknown, an axis is empty, or a scalar
            ``run_id`` is set.
    """
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    max_runs = int(data["max_runs"]) if "max_runs" in data else 0
    payload = {key: value for key, value in data.items() if key not in _SPACE_EXTRA}
    if "run_id" in payload and not isinstance(payload["run_id"], list):
        raise ValueError("omit scalar run_id; each trial uses a config digest")
    axes: dict[str, tuple[object, ...]] = {}
    fixed: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            if not value:
                raise ValueError(f"empty search axis {key!r}")
            axes[key] = tuple(value)
        else:
            fixed[key] = value
    names = tuple(sorted(axes))
    rows: list[dict[str, object]] = []
    if names:
        for combo in product(*(axes[name] for name in names)):
            row = dict(fixed)
            for name, item in zip(names, combo, strict=True):
                row[name] = item
            rows.append(row)
    else:
        rows.append(fixed)
    if max_runs > 0:
        rows = rows[:max_runs]
    trials = tuple(apply_train_mapping(TrainConfig(), row) for row in rows)
    for cfg in trials:
        resolve_arch(cfg.kind, cfg.arch)
    return trials


def completed_run_ids(jsonl: str | Path) -> frozenset[str]:
    """Return the run identifiers already recorded in a sweep JSONL.

    Args:
        jsonl: Path to a sweep JSONL. A missing file yields an empty set.

    Returns:
        frozenset[str]: Run identifiers of trials that finished without an
        error. Failed trials are excluded so a resume retries them.
    """
    path = Path(jsonl)
    if not path.is_file():
        return frozenset()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            continue
        run_id = record.get("run_id")
        if isinstance(run_id, str) and not record.get("error"):
            done.add(run_id)
    return frozenset(done)


def _trial_run_id(cfg: TrainConfig) -> str:
    """Return the run identifier ``train`` will use for a trial config."""
    if cfg.run_id:
        return cfg.run_id
    return f"{cfg.kind}-{resolve_arch(cfg.kind, cfg.arch)}-{cfg.seed}-{config_digest(cfg)}"


def _trial_record(
    cfg: TrainConfig,
    run_root: Path,
    summary: dict[str, object],
) -> dict[str, object]:
    """Return the JSONL record for one finished trial."""
    record: dict[str, object] = {
        "run_id": summary.get("run_id", run_root.name),
        "path": str(run_root),
        "n_params": summary.get("n_params"),
        "flops": summary.get("flops"),
        "train_seconds": summary.get("train_seconds"),
        "best_epoch": summary.get("best_epoch"),
        "stopped_early": summary.get("stopped_early"),
        "val_metric": summary.get("val_metric"),
        "best_val_metric": summary.get("best_val_metric"),
    }
    for field in _TRIAL_FIELDS:
        record[field] = getattr(cfg, field)
    record["arch"] = summary.get("arch", cfg.arch)
    for key, value in summary.items():
        if str(key).startswith("val_"):
            record[key] = value
    return record


def _relocate(
    trials: tuple[TrainConfig, ...], data_dir: str | None, run_dir: str | None
) -> tuple[TrainConfig, ...]:
    """Return trials with their data and run paths overridden."""
    overrides: dict[str, object] = {}
    if data_dir is not None:
        overrides["data_dir"] = data_dir
    if run_dir is not None:
        overrides["run_dir"] = run_dir
    if not overrides:
        return trials
    return tuple(apply_train_mapping(cfg, overrides) for cfg in trials)


@contextmanager
def _output_lock(jsonl: Path) -> Iterator[None]:
    """Hold an exclusive marker beside ``jsonl`` for the duration of a sweep.

    Args:
        jsonl: The sweep's output path.

    Yields:
        None: With the lock held.

    Raises:
        FileExistsError: If the marker already exists.

    Notes:
        Two sweeps pointed at one output interleave their JSONL rows and, worse,
        train into the same run directories, so each overwrites results the
        other is still reading. Nothing about the output file itself reveals
        that: the rows look well formed and only a repeated run identifier
        betrays the collision. The marker makes the second sweep refuse to start
        instead. It is removed on the way out, including after a failure, so
        only a killed process can leave one behind.
    """
    lock = jsonl.with_suffix(jsonl.suffix + ".lock")
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()} started={time.time():.0f}\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"another sweep is writing {jsonl}; if no sweep is running, delete {lock}"
        ) from exc
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def sweep(
    space_path: str | Path,
    out: str | None = None,
    resume: bool = False,
    data_dir: str | None = None,
    run_dir: str | None = None,
) -> Path:
    """Train each trial, score val, and append JSONL records.

    Args:
        space_path: Sweep space TOML.
        out: Optional JSONL path. None uses ``<run_dir>/sweep.jsonl``.
        resume: When true, append to an existing JSONL and skip the trials it
            already records. When false, start the JSONL from scratch.
        data_dir: Optional pack directory applied to every trial, overriding
            the space file. A committed space can then name a repository-
            relative default while a training box keeps its corpus elsewhere.
        run_dir: Optional run parent directory applied to every trial.

    Returns:
        Path: JSONL path.

    Raises:
        ValueError: If the space is invalid or a trial architecture is unknown.
        FileExistsError: If another sweep already holds the lock on ``out``.

    Notes:
        A trial that raises is recorded with an ``error`` field and the sweep
        continues; the run itself is not re-raised. Trials write their run
        directory with ``overwrite`` set, so a directory left behind by an
        interrupted trial is replaced rather than blocking the sweep.
    """
    trials = _relocate(load_sweep_space(space_path), data_dir, run_dir)
    if not trials:
        raise ValueError("sweep space produced no trials")
    jsonl = Path(out) if out is not None else Path(trials[0].run_dir) / "sweep.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = completed_run_ids(jsonl) if resume else frozenset()
    mode = "a" if resume else "w"
    with _output_lock(jsonl), jsonl.open(mode, encoding="utf-8") as handle:
        for cfg in trials:
            run_id = _trial_run_id(cfg)
            if run_id in done:
                continue
            trial = apply_train_mapping(cfg, {"overwrite": True})
            try:
                run_root = train(trial)
                evaluate(run_root, split="val")
                summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
                record = _trial_record(trial, run_root, summary)
            except (ValueError, RuntimeError, OSError) as exc:
                record = {"run_id": run_id, "kind": trial.kind, "arch": trial.arch}
                record["error"] = f"{type(exc).__name__}: {exc}"
            # An out-of-memory trial leaves its allocations cached, which would
            # push the next, otherwise-viable trial over the limit as well.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            handle.write(json.dumps(record) + "\n")
            handle.flush()
    return jsonl
