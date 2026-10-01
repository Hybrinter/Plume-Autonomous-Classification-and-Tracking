"""Discover, list, and compare inference run directories.

A run directory is any folder that contains ``summary.json``.

Contains:
  - discover_runs: sorted run paths under a parent directory.
  - load_summary: parse summary.json plus optional eval.json.
  - rank_runs: sort summaries by a val metric then FLOPs.
  - format_list / format_compare / format_rank: text tables for the CLI.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_COMPARE_KEYS: tuple[str, ...] = (
    "run_id",
    "kind",
    "arch",
    "best_epoch",
    "val_metric",
    "best_val_metric",
    "n_params",
    "flops",
    "val_f1",
    "val_mean_iou",
    "test_f1",
    "test_mean_iou",
    "test_n",
    "dataset_hash",
)


def discover_runs(root: str | Path) -> tuple[Path, ...]:
    """Return run directories under ``root`` that contain summary.json.

    Args:
        root: Parent directory, usually ``artifacts/runs``.

    Returns:
        tuple[Path, ...]: Sorted run paths.
    """
    parent = Path(root)
    if not parent.is_dir():
        return ()
    found: list[Path] = []
    for path in sorted(parent.iterdir()):
        if path.is_dir() and (path / "summary.json").is_file():
            found.append(path)
    return tuple(found)


def load_summary(run_dir: str | Path) -> dict[str, object]:
    """Load summary.json and overlay eval.json test fields when present.

    Args:
        run_dir: Run directory.

    Returns:
        dict[str, object]: Combined identity and metrics.

    Raises:
        FileNotFoundError: If summary.json is missing.
    """
    root = Path(run_dir)
    payload = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary.json must be an object")
    eval_path = root / "eval.json"
    if eval_path.is_file():
        eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
        if not isinstance(eval_payload, dict):
            raise ValueError("eval.json must be an object")
        split = str(eval_payload.get("split", "test"))
        prefix = "test_" if split == "test" else f"{split}_"
        for key, value in eval_payload.items():
            if key in {"split", "checkpoint", "kind", "arch"}:
                continue
            prefixed = f"{prefix}{key}"
            payload[prefixed] = value
    payload.setdefault("run_id", root.name)
    payload["path"] = str(root)
    return payload


def _cell(value: object) -> str:
    """Format one table cell."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def format_list(runs: tuple[Path, ...]) -> str:
    """Return a text table of discovered runs.

    Args:
        runs: Run directories.

    Returns:
        str: Header plus one row per run.
    """
    headers = (
        "run_id",
        "kind",
        "arch",
        "best_epoch",
        "best_val_metric",
        "n_params",
        "flops",
        "dataset_hash",
    )
    lines = ["\t".join(headers)]
    for path in runs:
        row = load_summary(path)
        lines.append("\t".join(_cell(row.get(key, "")) for key in headers))
    return "\n".join(lines) + "\n"


def format_compare(runs: tuple[Path, ...]) -> str:
    """Return a side-by-side table of selected summary and eval fields.

    Args:
        runs: Run directories to compare.

    Returns:
        str: Header plus one row per run.
    """
    lines = ["\t".join(_COMPARE_KEYS)]
    for path in runs:
        row = load_summary(path)
        lines.append("\t".join(_cell(row.get(key, "")) for key in _COMPARE_KEYS))
    return "\n".join(lines) + "\n"


def _metric_value(row: dict[str, object], metric: str) -> float:
    """Return the scalar used to rank ``row`` for ``metric``."""
    candidates: tuple[object, ...] = (
        row.get(f"val_{metric}"),
        row.get(metric),
        row.get("best_val_metric"),
    )
    for item in candidates:
        if isinstance(item, (int, float)):
            return float(item)
    return math.nan


def _flops_value(row: dict[str, object]) -> float:
    """Return FLOPs as a float, or +inf when missing."""
    item = row.get("flops")
    if isinstance(item, (int, float)):
        return float(item)
    return math.inf


def rank_runs(runs: tuple[Path, ...], metric: str) -> tuple[dict[str, object], ...]:
    """Sort run summaries by val metric, then by FLOPs ascending.

    Args:
        runs: Run directories.
        metric: Score name such as ``mean_iou``, ``f1``, or ``bce``.

    Returns:
        tuple[dict[str, object], ...]: Ranked ``load_summary`` rows. ``bce`` is
        minimized. Other metrics are maximized.
    """
    rows = tuple(load_summary(path) for path in runs)
    minimize = metric == "bce"

    def sort_key(row: dict[str, object]) -> tuple[float, float]:
        score = _metric_value(row, metric)
        if math.isnan(score):
            score = math.inf if minimize else -math.inf
        ordered = score if minimize else -score
        return (ordered, _flops_value(row))

    return tuple(sorted(rows, key=sort_key))


def format_rank(rows: tuple[dict[str, object], ...]) -> str:
    """Return a compare table in ranked order.

    Args:
        rows: Ranked summary dicts.

    Returns:
        str: Header plus one row per summary.
    """
    lines = ["\t".join(_COMPARE_KEYS)]
    for row in rows:
        lines.append("\t".join(_cell(row.get(key, "")) for key in _COMPARE_KEYS))
    return "\n".join(lines) + "\n"
