"""Discover, list, and compare inference run directories.

The implementation lives in ``tools.ml_models.analysis.runs``. This module
re-exports that catalog.

A run directory is any folder that contains ``summary.json``.

Contains:
  - discover_runs: sorted run paths under a parent directory.
  - load_summary: parse summary.json plus optional eval.json.
  - rank_runs: sort summaries by a val metric then FLOPs.
  - format_list / format_compare / format_rank: text tables for the CLI.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from tools.ml_models.analysis.runs import (
    discover_runs,
    format_compare,
    format_list,
    format_rank,
    load_summary,
    rank_runs,
)

__all__ = [
    "discover_runs",
    "format_compare",
    "format_list",
    "format_rank",
    "load_summary",
    "rank_runs",
]
