"""Tests for Normalize / Randomize / RandomCrop / ToTensor."""

from __future__ import annotations

import numpy as np
import torch

from smoke_detection.data.transforms import (
    CHANNEL_MEANS,
    CHANNEL_STDS,
    Normalize,
    RandomCrop,
    Randomize,
    ToTensor,
)

# ----------------------------- Normalize -----------------------------


def test_normalize_hand_computed():
    img = np.full((4, 4, 4), 1500.0, dtype=np.float32)
    out = Normalize()({"img": img.copy()})["img"]
    expected = (1500.0 - CHANNEL_MEANS.reshape(-1, 1, 1)) / CHANNEL_STDS.reshape(-1, 1, 1)
    expected = np.broadcast_to(expected, out.shape)
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_normalize_zero_centers_means():
    img = np.zeros((4, 3, 3), dtype=np.float32)
    img += CHANNEL_MEANS.reshape(-1, 1, 1)
    out = Normalize()({"img": img})["img"]
    np.testing.assert_allclose(out, np.zeros_like(out), atol=1e-5)


def test_normalize_broadcasts_over_spatial_dims():
    img = np.random.randn(4, 7, 11).astype(np.float32)
    out = Normalize()({"img": img.copy()})["img"]
    assert out.shape == img.shape


# ----------------------------- RandomCrop -----------------------------


def test_random_crop_produces_requested_shape():
    img = np.random.randn(4, 120, 120).astype(np.float32)
    out = RandomCrop(crop=90)({"img": img})["img"]
    assert out.shape == (4, 90, 90)


def test_random_crop_matches_fpt_offsets():
    rng = np.random.default_rng(0)
    img = rng.random((4, 120, 120)).astype(np.float32)
    fpt = rng.random((120, 120)).astype(np.float32)
    # Seed numpy to fix offsets
    np.random.seed(42)
    out = RandomCrop(crop=80)({"img": img.copy(), "fpt": fpt.copy()})
    # Reapply to verify we get the exact same crop when reseeded
    np.random.seed(42)
    out2 = RandomCrop(crop=80)({"img": img.copy(), "fpt": fpt.copy()})
    np.testing.assert_array_equal(out["img"], out2["img"])
    np.testing.assert_array_equal(out["fpt"], out2["fpt"])
    # Spatial dims must match between img and fpt
    assert out["img"].shape[-2:] == out["fpt"].shape


# ----------------------------- Randomize -----------------------------


def test_randomize_is_deterministic_under_seed():
    rng = np.random.default_rng(0)
    img = rng.random((4, 10, 10)).astype(np.float32)
    fpt = rng.random((10, 10)).astype(np.float32)
    np.random.seed(9)
    out_a = Randomize()({"img": img.copy(), "fpt": fpt.copy()})
    np.random.seed(9)
    out_b = Randomize()({"img": img.copy(), "fpt": fpt.copy()})
    np.testing.assert_array_equal(out_a["img"], out_b["img"])
    np.testing.assert_array_equal(out_a["fpt"], out_b["fpt"])


def test_randomize_flips_image_and_mask_together():
    # Place a single positive pixel at (0, 0) in the mask; check that after
    # the transform, the image pixel at the mask's positive location is the
    # same one the image had there originally.
    img = np.zeros((1, 4, 4), dtype=np.float32)
    img[0, 0, 0] = 42.0
    fpt = np.zeros((4, 4), dtype=np.float32)
    fpt[0, 0] = 1.0
    np.random.seed(0)
    out = Randomize()({"img": img, "fpt": fpt})
    i, j = np.argwhere(out["fpt"] == 1.0)[0]
    assert out["img"][0, i, j] == 42.0


# ----------------------------- ToTensor -----------------------------


def test_totensor_returns_float_tensors():
    sample = {
        "img": np.random.randn(4, 4, 4).astype(np.float32),
        "fpt": np.random.randn(4, 4).astype(np.float32),
    }
    out = ToTensor()(sample)
    assert isinstance(out["img"], torch.Tensor)
    assert out["img"].dtype == torch.float32
    assert isinstance(out["fpt"], torch.Tensor)
    assert out["fpt"].dtype == torch.float32


def test_totensor_without_fpt():
    sample = {"img": np.random.randn(4, 4, 4).astype(np.float32)}
    out = ToTensor()(sample)
    assert isinstance(out["img"], torch.Tensor)
    assert "fpt" not in out
