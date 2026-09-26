"""One synthetic ground-sample cell through scripts/gsd_sweep.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from tools.inference.data import load_processed_pack
from tools.ml_models.data.bands import ZENODO_BAND_IDS
from tools.ml_models.data.grid import resample_area
from tools.ml_models.data.matrix import Cell
from tools.ml_models.data.norm import apply_band_z, fit_band_stats


def _script() -> ModuleType:
    """Load scripts/gsd_sweep.py."""
    path = Path(__file__).resolve().parents[4] / "scripts" / "gsd_sweep.py"
    spec = importlib.util.spec_from_file_location("gsd_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_exits_zero() -> None:
    """--help exits 0."""
    module = _script()
    try:
        module.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_missing_cache_exits_without_download() -> None:
    """No arguments and no cache print a message and exit 2."""
    module = _script()
    code = module.main([])
    assert code == 2


def _normalized_subset(
    images: np.ndarray,
    channels: tuple[int, ...],
    side_px: int,
    train: tuple[int, ...],
) -> np.ndarray:
    """Resample selected channels and apply train-split band_z."""
    count = int(images.shape[0])
    picked = images[:, list(channels)]
    resized = np.empty((count, len(channels), side_px, side_px), dtype=np.float32)
    for index in range(count):
        resized[index] = resample_area(picked[index], side_px)
    stats = fit_band_stats(resized[list(train)])
    return apply_band_z(resized, stats)


def test_prepare_band_z_selects_subset_and_side(tmp_path: Path) -> None:
    """rgb at side 4 and s2_12 at side 6 are different packs."""
    module = _script()
    count = 2
    source_side = 8
    band_count = len(ZENODO_BAND_IDS)
    images = np.zeros((count, band_count, source_side, source_side), dtype=np.float32)
    for index in range(band_count):
        images[:, index] = float(index + 1)
        row = (index // 4) * 2
        col = (index % 4) * 2
        images[:, index, row : row + 2, col : col + 2] = float(100 + index)
    masks = np.zeros((count, source_side, source_side), dtype=np.float32)
    labels = np.zeros((count, 1), dtype=np.float32)
    src = tmp_path / "src"
    src.mkdir()
    np.save(src / "images.npy", images)
    np.save(src / "masks.npy", masks)
    np.save(src / "labels.npy", labels)
    (src / "splits.json").write_text(
        json.dumps({"train": [0], "val": [], "test": [1]}) + "\n",
        encoding="utf-8",
    )
    b2 = ZENODO_BAND_IDS.index("B2")
    assert b2 == 1

    rgb_dir = module.prepare_band_z_pack(src, tmp_path / "rgb", subset="rgb", side_px=4)
    rgb = np.load(rgb_dir / "images.npy")
    assert rgb.shape == (count, 3, 4, 4)
    expected = _normalized_subset(images, (b2, b2 + 1, b2 + 2), 4, (0,))
    np.testing.assert_allclose(rgb[:, 0], expected[:, 0], rtol=1e-5, atol=1e-5)
    wrong = _normalized_subset(images, (0, 1, 2), 4, (0,))
    assert not np.allclose(rgb, wrong)
    rgb_meta = json.loads((rgb_dir / "dataset.json").read_text(encoding="utf-8"))
    assert rgb_meta["n"] == count
    assert rgb_meta["height"] == 4
    assert rgb_meta["width"] == 4
    assert rgb_meta["in_channels"] == 3
    assert np.load(rgb_dir / "masks.npy").shape == (count, 1, 4, 4)

    wide_dir = module.prepare_band_z_pack(src, tmp_path / "s2", subset="s2_12", side_px=6)
    wide = np.load(wide_dir / "images.npy")
    assert wide.shape == (count, 12, 6, 6)
    assert rgb.shape != wide.shape
    wide_meta = json.loads((wide_dir / "dataset.json").read_text(encoding="utf-8"))
    assert wide_meta["height"] == 6
    assert wide_meta["width"] == 6
    assert wide_meta["in_channels"] == 12
    assert np.load(wide_dir / "masks.npy").shape == (count, 1, 6, 6)


def test_prepare_band_z_keeps_values_above_one(tmp_path: Path) -> None:
    """band_z planes above 1 survive the processed-pack loader."""
    module = _script()
    images = np.zeros((4, 1, 4, 4), dtype=np.float32)
    images[0] = 5.0
    masks = np.zeros((4, 1, 4, 4), dtype=np.float32)
    labels = np.zeros((4, 1), dtype=np.float32)
    src = tmp_path / "src"
    src.mkdir()
    np.save(src / "images.npy", images)
    np.save(src / "masks.npy", masks)
    np.save(src / "labels.npy", labels)
    (src / "splits.json").write_text(
        json.dumps({"train": [1, 2], "val": [3], "test": [0]}) + "\n",
        encoding="utf-8",
    )
    dest = module.prepare_band_z_pack(src, tmp_path / "z")
    pack = load_processed_pack(dest)
    stored = pack.images
    if isinstance(stored, np.ndarray):
        peak = float(np.max(stored))
    else:
        peak = float(np.max(stored.numpy()))
    assert peak > 1.0


def test_run_cell_trains_one_synthetic_cell_without_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One epoch and one step return a run directory and do not export."""

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("export called")

    import tools.inference.export as inference_export
    import tools.ml_models.export.onnx as onnx_export

    monkeypatch.setattr(onnx_export, "export", _boom)
    monkeypatch.setattr(inference_export, "export", _boom)

    from tools.inference.data import make_synthetic_pack, write_processed_pack
    from tools.inference.split import SplitRecipe

    module = _script()
    images, masks, labels = make_synthetic_pack(4, 3, 16, 16, seed=0)
    pack = tmp_path / "pack"
    write_processed_pack(pack, images, masks, labels, SplitRecipe(seed=0))
    cell = Cell(task="classify", subset="rgb", side_px=120)
    run_dir = module.run_cell(
        cell,
        pack,
        tmp_path / "out",
        epochs=1,
        max_steps=1,
    )
    assert run_dir.is_dir()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["arch"] == "shufflenetv2_x0_5"
    assert summary["loss"] == "bce"
    assert summary["kind"] == "classifier"
    eval_payload = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    assert eval_payload["split"] == "test"
