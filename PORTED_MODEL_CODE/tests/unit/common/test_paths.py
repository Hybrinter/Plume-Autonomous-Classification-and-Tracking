"""Tests for smoke_detection.common.paths."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from smoke_detection.common import paths as paths_module


def test_classification_split_joins_parts():
    base = Path("/tmp/fake")
    out = paths_module.classification_split("train", base)
    assert out == base / "classification" / "train"


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_classification_split_each_split(split):
    base = Path("/tmp/fake")
    out = paths_module.classification_split(split, base)
    assert out.name == split
    assert out.parent.name == "classification"


def test_segmentation_split_returns_image_and_label_dirs():
    base = Path("/tmp/fake")
    imgs, lbls = paths_module.segmentation_split("train", base)
    assert imgs == base / "segmentation" / "train" / "images"
    assert lbls == base / "segmentation" / "train" / "labels"


def test_explicit_root_overrides_default(tmp_path):
    out = paths_module.classification_split("train", tmp_path)
    assert out == tmp_path / "classification" / "train"


def test_env_var_overrides_dataset_root(monkeypatch, tmp_path):
    custom = tmp_path / "custom_root"
    monkeypatch.setenv("SMOKEDET_DATA_ROOT", str(custom))
    reloaded = importlib.reload(paths_module)
    try:
        assert custom.resolve() == reloaded.DATASET_ROOT
    finally:
        importlib.reload(paths_module)  # restore module-level state
