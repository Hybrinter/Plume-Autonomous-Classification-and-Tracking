"""Cartesian hyperparameter sweep over the local run catalog.

A space TOML uses TrainConfig field names. Scalar keys are fixed. List keys
are search axes. Trials expand in sorted-axis cartesian order and stop at
``max_runs`` when that key is set and positive. Each trial trains, scores the
val split, and appends one JSONL record.

Contains:
  - load_sweep_space: expand a space file into TrainConfig trials.
  - sweep: run trials and write sweep.jsonl.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
import tomllib
from itertools import product
from pathlib import Path

from tools.inference.arch.registry import resolve_arch
from tools.inference.eval import evaluate
from tools.inference.train import TrainConfig, apply_train_mapping, train

_SPACE_EXTRA = frozenset({"max_runs"})


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


def sweep(space_path: str | Path, out: str | None = None) -> Path:
    """Train each trial, score val, and append JSONL records.

    Args:
        space_path: Sweep space TOML.
        out: Optional JSONL path. None uses ``<run_dir>/sweep.jsonl``.

    Returns:
        Path: JSONL path.

    Raises:
        ValueError: If the space is invalid or a trial architecture is unknown.
        FileExistsError: If a trial run directory already exists.
    """
    trials = load_sweep_space(space_path)
    if not trials:
        raise ValueError("sweep space produced no trials")
    jsonl = Path(out) if out is not None else Path(trials[0].run_dir) / "sweep.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w", encoding="utf-8") as handle:
        for cfg in trials:
            run_root = train(cfg)
            evaluate(run_root, split="val")
            summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
            record: dict[str, object] = {
                "run_id": summary.get("run_id", run_root.name),
                "path": str(run_root),
                "kind": cfg.kind,
                "arch": summary.get("arch", cfg.arch),
                "learning_rate": cfg.learning_rate,
                "seed": cfg.seed,
                "optimizer": cfg.optimizer,
                "n_params": summary.get("n_params"),
                "flops": summary.get("flops"),
                "val_metric": summary.get("val_metric"),
                "best_val_metric": summary.get("best_val_metric"),
            }
            for key, value in summary.items():
                if str(key).startswith("val_"):
                    record[key] = value
            handle.write(json.dumps(record) + "\n")
    return jsonl
