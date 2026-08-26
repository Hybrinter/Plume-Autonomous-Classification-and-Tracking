"""Integration: scripts/prepare_dataset.py on a synthetic Zenodo-like source."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests._data import build_synthetic_zenodo_source

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "prepare_dataset.py"


def _run(*args, cwd=None, check=True):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def test_end_to_end_copy_mode(tmp_path):
    src = tmp_path / "source"
    build_synthetic_zenodo_source(src)
    out = tmp_path / "prepared"
    r = _run("--source", str(src), "--output", str(out), "--mode", "copy")
    assert r.returncode == 0
    total_tifs = len(list((out / "classification").rglob("*.tif")))
    assert total_tifs > 0


def test_splits_are_site_disjoint(tmp_path):
    src = tmp_path / "source"
    build_synthetic_zenodo_source(src)
    out = tmp_path / "prepared"
    _run("--source", str(src), "--output", str(out), "--mode", "copy")

    def _sites(root: Path) -> set[str]:
        return {p.name.split("_", 1)[0] for p in root.rglob("*.tif")}

    tr = _sites(out / "classification" / "train")
    va = _sites(out / "classification" / "val")
    te = _sites(out / "classification" / "test")
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)


def test_every_json_has_matching_positive_tif(tmp_path):
    src = tmp_path / "source"
    build_synthetic_zenodo_source(src)
    out = tmp_path / "prepared"
    _run("--source", str(src), "--output", str(out), "--mode", "copy")
    for split in ("train", "val", "test"):
        lbl_dir = out / "segmentation" / split / "labels"
        img_dir = out / "segmentation" / split / "images" / "positive"
        if not lbl_dir.exists():
            continue
        for jf in lbl_dir.glob("*.json"):
            # Label file is "<tif_basename>.json" — e.g. "site_001.tif.json",
            # so jf.stem is "site_001.tif" which is also the tif basename.
            candidate = img_dir / jf.stem
            assert candidate.exists(), f"{candidate} missing"


def test_dry_run_produces_no_output(tmp_path):
    src = tmp_path / "source"
    build_synthetic_zenodo_source(src)
    out = tmp_path / "prepared"
    r = _run("--source", str(src), "--output", str(out), "--dry-run")
    assert r.returncode == 0
    assert not out.exists()


def test_bad_ratios_fail(tmp_path):
    src = tmp_path / "source"
    build_synthetic_zenodo_source(src)
    out = tmp_path / "prepared"
    r = _run(
        "--source",
        str(src),
        "--output",
        str(out),
        "--mode",
        "copy",
        "--train-ratio",
        "0.9",
        "--val-ratio",
        "0.5",
        "--test-ratio",
        "0.1",
        check=False,
    )
    assert r.returncode != 0
    assert "ratios must sum to 1.0" in r.stderr


def test_missing_source_fails(tmp_path):
    out = tmp_path / "prepared"
    r = _run(
        "--source",
        str(tmp_path / "does_not_exist"),
        "--output",
        str(out),
        "--mode",
        "copy",
        check=False,
    )
    assert r.returncode != 0


def test_refuses_existing_output_without_force(tmp_path):
    src = tmp_path / "source"
    build_synthetic_zenodo_source(src)
    out = tmp_path / "prepared"
    out.mkdir()
    r = _run("--source", str(src), "--output", str(out), "--mode", "copy", check=False)
    assert r.returncode != 0
    assert "--force" in r.stderr
