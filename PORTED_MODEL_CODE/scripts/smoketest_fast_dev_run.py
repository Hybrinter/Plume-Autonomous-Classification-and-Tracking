"""End-to-end smoke test: run Lightning fast_dev_run on a synthetic dataset.

Creates a tiny prepared-dataset tree, points SMOKEDET_DATA_ROOT at it, and
runs training for one batch on both the classification and segmentation
configs. Exits non-zero on any exception.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.transform import from_origin


def _write_fake_tif(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (np.random.rand(13, 120, 120) * 3000).astype(np.uint16)
    profile = {
        "driver": "GTiff",
        "width": 120,
        "height": 120,
        "count": 13,
        "dtype": "uint16",
        "transform": from_origin(0, 0, 1, 1),
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(data)


def _write_fake_label(path: Path, tif_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completions": [
            {"result": [{"value": {"points": [[10, 10], [60, 10], [60, 60], [10, 60]]}}]}
        ],
        "data": {"image": f"/data/upload/1-{tif_name.replace('.tif', '.png')}"},
    }
    path.write_text(json.dumps(payload))


def _build_fake_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        for cls in ("positive", "negative"):
            for i in range(4):
                fname = f"{cls}_{split}_{i:03d}.tif"
                _write_fake_tif(root / "classification" / split / cls / fname)
                _write_fake_tif(root / "segmentation" / split / "images" / cls / fname)
                if cls == "positive":
                    _write_fake_label(
                        root / "segmentation" / split / "labels" / f"{fname}.json", fname
                    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="smokedet_smoke_"))
    try:
        _build_fake_dataset(tmp)
        os.environ["SMOKEDET_DATA_ROOT"] = str(tmp)
        from smoke_detection.cli.train import main as train_main

        print(f"[smoketest] data root: {tmp}")
        print("[smoketest] running classification fast_dev_run…")
        rc = train_main(
            [
                "--config",
                "configs/classification/default.yaml",
                "--override",
                f"paths.data_root={tmp}",
                "--override",
                "trainer.fast_dev_run=true",
                "--override",
                "data.batch_size=2",
                "--override",
                "data.num_workers=0",
            ]
        )
        if rc != 0:
            return rc

        print("[smoketest] running segmentation fast_dev_run…")
        rc = train_main(
            [
                "--config",
                "configs/segmentation/default.yaml",
                "--override",
                f"paths.data_root={tmp}",
                "--override",
                "trainer.fast_dev_run=true",
                "--override",
                "data.batch_size=2",
                "--override",
                "data.num_workers=0",
            ]
        )
        return rc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
