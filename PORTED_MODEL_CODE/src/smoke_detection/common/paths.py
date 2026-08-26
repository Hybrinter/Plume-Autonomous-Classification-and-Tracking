"""Filesystem paths for the prepared dataset.

Default layout (created by ``scripts/prepare_dataset.py``):

    <repo_root>/data/
        classification/{train,val,test}/{positive,negative}/*.tif
        segmentation/{train,val,test}/images/{positive,negative}/*.tif
        segmentation/{train,val,test}/labels/*.json

Override the root via ``SMOKEDET_DATA_ROOT`` env var or the ``paths.data_root``
YAML field.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/smoke_detection/common/paths.py -> src/smoke_detection/common ->
# src/smoke_detection -> src -> repo_root
_REPO_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT: Path = Path(os.environ.get("SMOKEDET_DATA_ROOT", str(_REPO_ROOT / "data"))).resolve()


def classification_split(name: str, root: Path | None = None) -> Path:
    """Return the classification split directory for ``name`` in
    (``"train"``, ``"val"``, ``"test"``)."""
    base = (root or DATASET_ROOT) / "classification" / name
    return base


def segmentation_split(name: str, root: Path | None = None) -> tuple[Path, Path]:
    """Return ``(images_dir, labels_dir)`` for the segmentation split ``name``."""
    base = (root or DATASET_ROOT) / "segmentation" / name
    return base / "images", base / "labels"
