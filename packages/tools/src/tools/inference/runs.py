"""Discover, list, and compare inference run directories.

A run directory is any folder that contains ``summary.json``.

Contains:
  - discover_runs: sorted run paths under a parent directory.
  - load_summary: parse summary.json plus optional eval.json.
  - format_list / format_compare: text tables for the CLI.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
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
