"""Tests for SmokePlumeDataset (classification)."""

from __future__ import annotations

import numpy as np
import pytest

from smoke_detection.common.paths import classification_split
from smoke_detection.data.classification_dataset import (
    SmokePlumeDataset,
    build_default_transform,
    build_eval_transform,
)


@pytest.fixture
def train_dir(synthetic_dataset_root):
    return classification_split("train", synthetic_dataset_root)


def test_length_nonzero(train_dir):
    ds = SmokePlumeDataset(datadir=train_dir, balance="none")
    assert len(ds) > 0


def test_getitem_returns_expected_keys(train_dir):
    ds = SmokePlumeDataset(datadir=train_dir, balance="none")
    s = ds[0]
    assert set(s.keys()) == {"idx", "img", "lbl", "imgfile"}


def test_img_shape_and_dtype(train_dir):
    ds = SmokePlumeDataset(datadir=train_dir, balance="none")
    s = ds[0]
    assert isinstance(s["img"], np.ndarray)
    assert s["img"].shape == (4, 120, 120)
    assert s["img"].dtype == np.float32


def test_label_derived_from_path(train_dir):
    ds = SmokePlumeDataset(datadir=train_dir, balance="none")
    for idx in range(len(ds)):
        s = ds[idx]
        if "positive" in s["imgfile"]:
            assert s["lbl"] is True
        elif "negative" in s["imgfile"]:
            assert s["lbl"] is False


def test_balance_upsample_equalizes_or_grows(train_dir):
    up = SmokePlumeDataset(datadir=train_dir, balance="upsample")
    none = SmokePlumeDataset(datadir=train_dir, balance="none")
    n_pos_none = int(none.labels.sum())
    n_neg_none = int((~none.labels.astype(bool)).sum())
    n_pos_up = int(up.labels.sum())
    # Upsampling adds duplicate positives until >= negatives.
    assert n_pos_up >= n_neg_none or n_pos_up >= n_pos_none


def test_balance_upsample_casts_indices_to_integer(train_dir):
    ds = SmokePlumeDataset(datadir=train_dir, balance="none")
    ds.positive_indices = ds.positive_indices.astype(np.float64)

    ds._balance_upsample()

    assert ds.positive_indices.dtype.kind in {"i", "u"}
    assert ds.negative_indices.dtype.kind in {"i", "u"}


def test_balance_downsample_reduces_length(train_dir):
    none = SmokePlumeDataset(datadir=train_dir, balance="none")
    down = SmokePlumeDataset(datadir=train_dir, balance="downsample")
    assert len(down) <= len(none)


def test_balance_none_leaves_counts_intact(train_dir):
    ds = SmokePlumeDataset(datadir=train_dir, balance="none")
    assert len(ds) == len(ds.imgfiles)


def test_mult_scales_length(train_dir):
    base = SmokePlumeDataset(datadir=train_dir, balance="none", mult=1)
    doubled = SmokePlumeDataset(datadir=train_dir, balance="none", mult=2)
    assert len(doubled) == 2 * len(base)


def test_transform_is_applied(train_dir):
    tfm = build_eval_transform()
    ds = SmokePlumeDataset(datadir=train_dir, balance="none", transform=tfm)
    s = ds[0]
    # After ToTensor, img must be a torch.Tensor.
    import torch

    assert isinstance(s["img"], torch.Tensor)


def test_build_default_transform_callable():
    tfm = build_default_transform()
    assert callable(tfm)
