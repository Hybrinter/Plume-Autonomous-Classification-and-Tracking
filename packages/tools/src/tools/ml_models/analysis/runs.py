"""Run-catalog boundary and text formatters for model runs.

The legacy discovery, summary reader, and ranking relied on artifact layouts
that standard training no longer writes. Those boundaries are unavailable
until the evidence analysis phase replaces them; they return explicit errors
rather than empty catalogs or fabricated rows. The formatters remain pure:
they render rows the caller already holds and never read files.

Contains:
  - discover_runs: unavailable run-directory catalog reader.
  - load_summary: unavailable run-summary reader.
  - rank_runs: unavailable stored-summary ranking.
  - format_list / format_compare / format_rank: text tables over given rows.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from pathlib import Path

from flight.libs.types import Err, Result

_UNAVAILABLE = "run catalog reading is unavailable until the evidence analysis phase is implemented"

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


def discover_runs(root: str | Path) -> Result[tuple[Path, ...], str]:
    """Refuse to discover runs while the catalog reader is unimplemented.

    Args:
        root: Parent directory, usually ``artifacts/runs``.

    Returns:
        Result[tuple[Path, ...], str]: Always Err.
    """
    del root
    return Err(_UNAVAILABLE)


def load_summary(run_dir: str | Path) -> Result[dict[str, object], str]:
    """Refuse to load a run summary while the catalog reader is unimplemented.

    Args:
        run_dir: Run directory.

    Returns:
        Result[dict[str, object], str]: Always Err.
    """
    del run_dir
    return Err(_UNAVAILABLE)


def rank_runs(runs: tuple[Path, ...], metric: str) -> Result[tuple[dict[str, object], ...], str]:
    """Refuse to rank stored run summaries while the reader is unimplemented.

    Args:
        runs: Run directories.
        metric: Score name such as ``mean_iou``, ``f1``, or ``bce``.

    Returns:
        Result[tuple[dict[str, object], ...], str]: Always Err.
    """
    del runs, metric
    return Err(_UNAVAILABLE)


def _cell(value: object) -> str:
    """Format one table cell."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def format_list(rows: tuple[dict[str, object], ...]) -> str:
    """Return a text table of already-loaded summary rows.

    Args:
        rows: Summary dicts, one per run.

    Returns:
        str: Header plus one row per summary.
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
    for row in rows:
        lines.append("\t".join(_cell(row.get(key, "")) for key in headers))
    return "\n".join(lines) + "\n"


def format_compare(rows: tuple[dict[str, object], ...]) -> str:
    """Return a side-by-side table of already-loaded summary rows.

    Args:
        rows: Summary dicts, one per run.

    Returns:
        str: Header plus one row per summary.
    """
    lines = ["\t".join(_COMPARE_KEYS)]
    for row in rows:
        lines.append("\t".join(_cell(row.get(key, "")) for key in _COMPARE_KEYS))
    return "\n".join(lines) + "\n"


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
