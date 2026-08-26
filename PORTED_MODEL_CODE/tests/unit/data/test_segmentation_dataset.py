"""Tests for SmokePlumeSegmentationDataset and label_image_url_to_tif_key."""

from __future__ import annotations

import numpy as np
import pytest

from smoke_detection.common.paths import segmentation_split
from smoke_detection.data.segmentation_dataset import (
    SmokePlumeSegmentationDataset,
    build_default_transform,
    build_eval_transform,
    label_image_url_to_tif_key,
)

# ------------------------- url helper -------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("/data/upload/1-site_001.png", "site_001.tif"),
        ("/data/upload/1-site_2024:01:01.png", "site_2024_01_01.tif"),
        ("/x/y-a-b.png", "a-b.tif"),
    ],
)
def test_label_image_url_to_tif_key(url, expected):
    assert label_image_url_to_tif_key(url) == expected


# ------------------------- dataset core -------------------------


@pytest.fixture
def train_dirs(synthetic_dataset_root):
    imgs, lbls = segmentation_split("train", synthetic_dataset_root)
    return imgs, lbls


def test_length_is_2x_positive_count(train_dirs):
    ds = SmokePlumeSegmentationDataset(datadir=train_dirs[0], seglabeldir=train_dirs[1])
    n_pos = int(ds.labels.sum())
    assert len(ds) == 2 * n_pos


def test_getitem_keys_and_shapes(train_dirs):
    ds = SmokePlumeSegmentationDataset(datadir=train_dirs[0], seglabeldir=train_dirs[1])
    s = ds[0]
    assert set(s.keys()) == {"idx", "img", "fpt", "imgfile"}
    assert s["img"].shape == (4, 120, 120)
    assert s["img"].dtype == np.float32
    assert s["fpt"].shape == (120, 120)
    assert s["fpt"].dtype == np.float32


def test_positive_has_nonzero_mask(train_dirs):
    ds = SmokePlumeSegmentationDataset(datadir=train_dirs[0], seglabeldir=train_dirs[1])
    # Positives are first (indices 0..n_pos-1)
    s = ds[0]
    assert s["fpt"].sum() > 0


def test_negative_has_zero_mask(train_dirs):
    ds = SmokePlumeSegmentationDataset(datadir=train_dirs[0], seglabeldir=train_dirs[1])
    n_pos = int(ds.labels.sum())
    s = ds[n_pos]  # first negative
    assert s["fpt"].sum() == 0


def test_polygon_scaled_by_1_2(train_dirs):
    """Synthetic polygon is a 50x50 square with corners [10,10], [60,10], [60,60], [10,60].
    After the 1.2x scale applied inside the dataset, corners become [12,12] .. [72,72],
    so the rasterized area is approximately 60x60 = 3600 px (± edge effects)."""
    ds = SmokePlumeSegmentationDataset(datadir=train_dirs[0], seglabeldir=train_dirs[1])
    s = ds[0]
    area = float(s["fpt"].sum())
    # Allow ±10% for rasterization edge effects.
    expected = 60 * 60
    assert 0.9 * expected <= area <= 1.1 * expected + 120  # generous due to all_touched=True


def test_mult_scales_length(train_dirs):
    base = SmokePlumeSegmentationDataset(datadir=train_dirs[0], seglabeldir=train_dirs[1], mult=1)
    doubled = SmokePlumeSegmentationDataset(
        datadir=train_dirs[0], seglabeldir=train_dirs[1], mult=2
    )
    assert len(doubled) == 2 * len(base)


def test_build_default_and_eval_transforms_callable():
    assert callable(build_default_transform())
    assert callable(build_eval_transform())
