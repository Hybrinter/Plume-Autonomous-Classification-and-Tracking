"""scripts/full_frame_eval.py on a tiny pack."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np


def _script() -> ModuleType:
    """Load scripts/full_frame_eval.py."""
    path = Path(__file__).resolve().parents[4] / "scripts" / "full_frame_eval.py"
    spec = importlib.util.spec_from_file_location("full_frame_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pack(root: Path) -> Path:
    """Write a tiny pack with a background chip and a plume on the test split."""
    side = 8
    images = np.full((4, 3, side, side), 0.15, dtype=np.float32)
    masks = np.zeros((4, 1, side, side), dtype=np.float32)
    labels = np.zeros((4, 1), dtype=np.float32)
    masks[3, 0, 1:7, 1:7] = 1.0
    labels[3, 0] = 1.0
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "images.npy", images)
    np.save(root / "masks.npy", masks)
    np.save(root / "labels.npy", labels)
    (root / "splits.json").write_text(
        json.dumps({"train": [0], "val": [1], "test": [2, 3]}) + "\n",
        encoding="utf-8",
    )
    return root


def test_help_exits_zero() -> None:
    """--help exits 0."""
    module = _script()
    try:
        module.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_dry_run_writes_summary_on_a_tiny_frame(tmp_path: Path) -> None:
    """--dry-run scores one tiny frame and does not use chip IoU as pass/fail."""
    module = _script()
    pack = _pack(tmp_path / "pack")
    out = tmp_path / "summary.json"
    code = module.main(
        [
            "--pack",
            str(pack),
            "--out",
            str(out),
            "--frame-h",
            "24",
            "--frame-w",
            "24",
            "--limit",
            "1",
            "--dry-run",
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "hit_rate_by_placement" in payload
    assert "empty_frame_false_positive_rate" in payload
    assert "chip_vs_frame_logit_margin" in payload
    assert "full_frame_hit_rate" in payload
    assert "chip_iou" in payload
    assert "passed" not in payload
    assert "pass" not in payload
