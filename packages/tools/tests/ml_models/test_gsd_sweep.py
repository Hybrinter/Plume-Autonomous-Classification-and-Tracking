"""One synthetic ground-sample cell through scripts/gsd_sweep.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from tools.ml_models.data.matrix import Cell
from tools.ml_models.train.samples import load_processed_pack


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
    loaded = pack.images
    peak = float(np.max(loaded)) if isinstance(loaded, np.ndarray) else float(loaded.numpy().max())
    assert peak > 1.0


def test_run_cell_trains_one_synthetic_cell_without_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One epoch and one step return a run directory and do not export."""

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("export called")

    import tools.ml_models.export.onnx as onnx_export

    monkeypatch.setattr(onnx_export, "export", _boom)

    from tools.ml_models.train.recipe import SplitRecipe
    from tools.ml_models.train.samples import make_synthetic_pack, write_processed_pack

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
