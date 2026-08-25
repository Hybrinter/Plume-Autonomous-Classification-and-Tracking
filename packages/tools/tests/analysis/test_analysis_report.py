"""Tests for the static report bundle emitter (report) and the CLI."""

import json
from pathlib import Path

import pandas as pd
from tools.analysis.cli import main
from tools.analysis.report import write_run_report, write_suite_report
from tools.analysis.runner import ScenarioSpec, run_scenario


def _small_spec(name: str = "test_small") -> ScenarioSpec:
    """A tiny nominal scenario for fast bundle tests."""
    return ScenarioSpec(
        name=name,
        title="Small test run",
        description="A short nominal run used by the report tests.",
        category="nominal",
        steps=4,
        num_frames=4,
    )


def test_run_report_emits_full_bundle(tmp_path: Path) -> None:
    """A run bundle has data (long/stats/wide parquet+csv), figures, summary, and manifest."""
    run = run_scenario(_small_spec())
    report = write_run_report(run, tmp_path)
    for relative in (
        "data/long.parquet",
        "data/long.csv",
        "data/stats.parquet",
        "data/stats.csv",
        "data/wide/payload.parquet",
        "data/wide/bus.csv",
        "summary.md",
        "summary.html",
        "manifest.json",
    ):
        assert (tmp_path / relative).is_file(), relative
    pngs = list((tmp_path / "figures").rglob("*.png"))
    assert len(pngs) == report.n_figures > 0
    assert all(path.stat().st_size > 0 for path in pngs)


def test_manifest_is_valid_and_complete(tmp_path: Path) -> None:
    """The manifest parses and carries the datapoint counts, groups, and outcomes."""
    run = run_scenario(_small_spec())
    write_run_report(run, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["deterministic"] is True
    assert manifest["datapoints"]["total_columns"] == run.capture.n_columns
    assert set(manifest["groups"]) == set(run.capture.wide)
    assert "outcomes" in manifest and "final_gimbal_state" in manifest["outcomes"]


def test_long_parquet_roundtrips(tmp_path: Path) -> None:
    """The emitted long Parquet reads back to the same shape as the captured long frame."""
    run = run_scenario(_small_spec())
    write_run_report(run, tmp_path)
    restored = pd.read_parquet(tmp_path / "data" / "long.parquet", engine="pyarrow")
    assert restored.shape == run.capture.long.shape


def test_run_report_is_deterministic(tmp_path: Path) -> None:
    """Two reports of the same run produce identical CSV datasets."""
    run = run_scenario(_small_spec())
    write_run_report(run, tmp_path / "a")
    write_run_report(run, tmp_path / "b")
    a = (tmp_path / "a" / "data" / "long.csv").read_text(encoding="utf-8")
    b = (tmp_path / "b" / "data" / "long.csv").read_text(encoding="utf-8")
    assert a == b


def test_write_suite_report_emits_index(tmp_path: Path) -> None:
    """A suite report writes an index (md/html/manifest) plus each run's bundle."""
    runs = [run_scenario(_small_spec("run_a")), run_scenario(_small_spec("run_b"))]
    suite = write_suite_report(runs, "test_suite", tmp_path)
    assert (tmp_path / "index.md").is_file()
    assert (tmp_path / "index.html").is_file()
    index = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert index["n_runs"] == 2
    assert (tmp_path / "run_a" / "summary.md").is_file()
    assert (tmp_path / "run_b" / "manifest.json").is_file()
    assert len(suite.runs) == 2


def test_cli_run_writes_bundle(tmp_path: Path) -> None:
    """The CLI 'run' command captures a scenario and writes a bundle to --out."""
    out = tmp_path / "bundle"
    code = main(["run", "command_ingress_auth", "--out", str(out)])
    assert code == 0
    assert (out / "command_ingress_auth" / "summary.html").is_file()
    assert (out / "manifest.json").is_file()


def test_cli_list_returns_zero() -> None:
    """The CLI 'list' command succeeds."""
    assert main(["list"]) == 0
