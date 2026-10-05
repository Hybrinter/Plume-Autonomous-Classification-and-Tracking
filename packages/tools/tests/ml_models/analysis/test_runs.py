"""Tests for the unavailable run-catalog boundary and retained formatters."""

import json
from pathlib import Path

import tools.ml_models.analysis.runs as runs
from flight.libs.types import Err
from tools.ml_models.analysis.runs import (
    discover_runs,
    format_compare,
    format_list,
    format_rank,
    load_summary,
    rank_runs,
)


def _write_run(root: Path, name: str, iou: float) -> Path:
    """Write a legacy summary.json run directory."""
    dest = root / name
    dest.mkdir()
    payload = {
        "run_id": name,
        "kind": "segmentor",
        "arch": "unet",
        "best_epoch": 1,
        "val_metric": "mean_iou",
        "best_val_metric": iou,
        "n_params": 1000,
        "flops": 2000,
        "val_mean_iou": iou,
        "dataset_hash": "abc",
    }
    (dest / "summary.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return dest


def test_catalog_readers_return_unavailable(tmp_path: Path) -> None:
    """Discovery, summary loading, and ranking fail closed on any input."""
    dest = _write_run(tmp_path, "a", 0.5)
    for result in (
        discover_runs(tmp_path),
        load_summary(dest),
        rank_runs((dest,), "mean_iou"),
    ):
        assert isinstance(result, Err)
        assert "unavailable" in result.error


def test_format_list_and_compare_render_given_rows() -> None:
    """The list and compare formatters render already-loaded row dicts."""
    rows = (
        {
            "run_id": "a",
            "kind": "segmentor",
            "arch": "unet",
            "best_epoch": 1,
            "best_val_metric": 0.5,
            "n_params": 1000,
            "flops": 2000,
            "dataset_hash": "abc",
        },
    )
    assert "a" in format_list(rows)
    table = format_compare(rows)
    assert "0.5" in table
    assert "1000" in table
    assert "2000" in table


def test_no_legacy_sort_fallback_remains() -> None:
    """The removed ranking fallback has no renamed helper left behind."""
    assert not hasattr(runs, "sort_rows")


def test_format_rank_renders_ranked_rows() -> None:
    """format_rank prints the compare table in the caller's order."""
    rows = (
        {"run_id": "a", "kind": "segmentor", "best_val_metric": 0.5, "flops": 2000},
        {"run_id": "b", "kind": "segmentor", "best_val_metric": 0.7, "flops": 1500},
    )
    table = format_rank(rows)
    lines = table.splitlines()
    assert lines[1].startswith("a\t")
    assert lines[2].startswith("b\t")
