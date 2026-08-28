"""Tests for local run discovery and compare tables."""

import json
from pathlib import Path

from tools.inference.cli import main
from tools.inference.runs import discover_runs, format_compare, format_list, load_summary


def _write_run(root: Path, name: str, kind: str, iou: float) -> Path:
    """Write a minimal summary.json run directory."""
    dest = root / name
    dest.mkdir()
    payload = {
        "run_id": name,
        "kind": kind,
        "arch": "unet",
        "best_epoch": 1,
        "val_metric": "mean_iou",
        "best_val_metric": iou,
        "dataset_hash": "abc",
    }
    (dest / "summary.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return dest


def test_discover_and_list(tmp_path: Path) -> None:
    """discover_runs finds summary.json folders and format_list prints them."""
    _write_run(tmp_path, "a", "segmentor", 0.5)
    _write_run(tmp_path, "b", "segmentor", 0.7)
    (tmp_path / "not-a-run").mkdir()
    runs = discover_runs(tmp_path)
    assert [path.name for path in runs] == ["a", "b"]
    table = format_list(runs)
    assert "a" in table
    assert "b" in table


def test_compare_includes_eval_overlay(tmp_path: Path) -> None:
    """format_compare shows test metrics from eval.json."""
    dest = _write_run(tmp_path, "exp", "segmentor", 0.4)
    (dest / "eval.json").write_text(
        json.dumps({"split": "test", "n": 3, "mean_iou": 0.6, "kind": "segmentor"}) + "\n",
        encoding="utf-8",
    )
    row = load_summary(dest)
    assert row["test_mean_iou"] == 0.6
    table = format_compare((dest,))
    assert "0.6" in table


def test_cli_list_and_compare(tmp_path: Path) -> None:
    """CLI list and compare print tables and return 0."""
    dest = _write_run(tmp_path, "cli-run", "classifier", 0.9)
    assert main(["list", "--run-dir", str(tmp_path)]) == 0
    assert main(["compare", "--run", str(dest)]) == 0
