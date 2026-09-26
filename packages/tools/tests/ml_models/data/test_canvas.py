"""Flight-frame canvas on a tiny grid. No 1544 by 2064 allocation."""

from __future__ import annotations

import numpy as np
import pytest
from tools.ml_models.data.canvas import (
    CanvasConfig,
    Chip,
    build_scene,
    sample_view,
    take_window,
)


def _config(*, empty_fraction: float = 0.0, feather_px: int = 2) -> CanvasConfig:
    """Return a tiny canvas config."""
    return CanvasConfig(
        frame_hw=(20, 24),
        window_px=12,
        chip_side=8,
        empty_fraction=empty_fraction,
        max_plumes=1,
        feather_px=feather_px,
        full_frame_every=8,
        seed=0,
    )


def _chip(
    *,
    label: float,
    split: str,
    annotated: bool,
    fill: float = 0.0,
    mask_fill: float = 0.0,
    side: int = 8,
) -> Chip:
    """Return one synthetic chip."""
    image = np.full((1, side, side), fill, dtype=np.float32)
    mask = np.full((1, side, side), mask_fill, dtype=np.float32)
    return Chip(
        image=image,
        mask=mask,
        label=label,
        group_id="loc-" + split,
        split=split,
        annotated=annotated,
    )


def test_empty_scene_mask_is_zero() -> None:
    """An empty scene and its window have an all-zero mask."""
    config = _config(empty_fraction=1.0)
    background = _chip(label=0.0, split="train", annotated=False)
    image, mask, label = build_scene((background,), config, np.random.default_rng(0))
    assert image.shape == (1, 20, 24)
    assert mask.shape == (1, 20, 24)
    assert np.all(mask == 0.0)
    assert label == 0.0
    sample = sample_view((background,), config, np.random.default_rng(1), full_frame=False)
    assert sample.full_frame is False
    assert sample.mask.shape == (1, 12, 12)
    assert np.all(sample.mask == 0.0)
    assert sample.label == 0.0


def test_plume_is_not_forced_to_the_window_center() -> None:
    """Across seeds, the plume centroid is not always the window center."""
    config = _config(empty_fraction=0.0, feather_px=0)
    background = _chip(label=0.0, split="train", annotated=False)
    positive = _chip(label=1.0, split="train", annotated=True, fill=1.0, mask_fill=1.0)
    center = (config.window_px - 1) / 2.0
    centroids: list[tuple[float, float]] = []
    differs = False
    for seed in range(16):
        sample = sample_view(
            (background, positive),
            config,
            np.random.default_rng(seed),
            full_frame=False,
        )
        assert sample.origin is not None
        assert sample.mask.shape == (1, config.window_px, config.window_px)
        assert float(sample.mask.max()) == 1.0
        rows, cols = np.nonzero(sample.mask[0] > 0.0)
        cy = float(rows.mean())
        cx = float(cols.mean())
        centroids.append((cy, cx))
        if abs(cy - center) > 1e-3 or abs(cx - center) > 1e-3:
            differs = True
    assert differs
    assert len(set(centroids)) > 1


def test_feather_does_not_change_mask_support() -> None:
    """Polygon pixels in frame stay 1, and the mask stays binary."""
    config = _config(empty_fraction=0.0, feather_px=2)
    background = _chip(label=0.0, split="train", annotated=False, fill=0.0)
    positive = _chip(label=1.0, split="train", annotated=True, fill=1.0, mask_fill=1.0)
    saw_feather = False
    for seed in range(8):
        image, mask, label = build_scene(
            (background, positive),
            config,
            np.random.default_rng(seed),
        )
        assert label == 1.0
        assert np.all((mask == 0.0) | (mask == 1.0))
        footprint = image[0] > 0.0
        assert np.array_equal(mask[0] > 0.0, footprint)
        blended = (image[0] > 0.0) & (image[0] < 1.0)
        if np.any(blended):
            saw_feather = True
            assert np.all(mask[0][blended] == 1.0)
            assert np.all(mask[0][~footprint] == 0.0)
    assert saw_feather


def test_interior_polygon_stays_in_a_nonempty_scene() -> None:
    """A plume inset from the chip border still marks a non-empty scene."""
    config = _config(empty_fraction=0.0, feather_px=0)
    background = _chip(label=0.0, split="train", annotated=False)
    positive = _chip(label=1.0, split="train", annotated=True, fill=1.0, mask_fill=0.0)
    positive.mask[0, 3:5, 3:5] = 1.0
    assert float(positive.mask[0, 0, :].max()) == 0.0
    assert float(positive.mask[0, -1, :].max()) == 0.0
    assert float(positive.mask[0, :, 0].max()) == 0.0
    assert float(positive.mask[0, :, -1].max()) == 0.0
    for seed in range(32):
        _image, mask, label = build_scene(
            (background, positive),
            config,
            np.random.default_rng(seed),
        )
        assert label == 1.0
        assert int(np.count_nonzero(mask > 0.0)) >= 1
    empty = _config(empty_fraction=1.0, feather_px=0)
    _image, mask, label = build_scene(
        (background, positive),
        empty,
        np.random.default_rng(0),
    )
    assert label == 0.0
    assert np.all(mask == 0.0)


def test_positive_split_must_match_background() -> None:
    """A positive chip from another split is refused."""
    config = _config(empty_fraction=1.0)
    background = _chip(label=0.0, split="train", annotated=False)
    positive = _chip(label=1.0, split="val", annotated=True, fill=1.0, mask_fill=1.0)
    with pytest.raises(ValueError, match="split"):
        build_scene((background, positive), config, np.random.default_rng(0))


def test_unannotated_positive_cannot_be_pasted() -> None:
    """An unannotated positive raises when the scene would paste it."""
    config = _config(empty_fraction=0.0)
    background = _chip(label=0.0, split="train", annotated=False)
    positive = _chip(label=1.0, split="train", annotated=False, fill=1.0, mask_fill=1.0)
    with pytest.raises(ValueError, match="unannotated"):
        build_scene((background, positive), config, np.random.default_rng(0))


def test_window_crop_matches_the_scene_slice() -> None:
    """The reported origin addresses the same pixels as the scene."""
    config = _config(empty_fraction=0.0, feather_px=0)
    background = _chip(label=0.0, split="train", annotated=False)
    positive = _chip(label=1.0, split="train", annotated=True, fill=1.0, mask_fill=1.0)
    rng = np.random.default_rng(3)
    image, mask, _label = build_scene((background, positive), config, rng)
    crop, crop_mask, origin = take_window(image, mask, config, rng)
    top, left = origin
    window = config.window_px
    assert np.array_equal(crop, image[:, top : top + window, left : left + window])
    assert np.array_equal(crop_mask, mask[:, top : top + window, left : left + window])
    assert float(crop_mask.max()) == 1.0


def test_full_frame_view_keeps_the_scene() -> None:
    """A full-frame sample has no window origin and matches the scene shape."""
    config = _config(empty_fraction=1.0)
    background = _chip(label=0.0, split="train", annotated=False)
    sample = sample_view((background,), config, np.random.default_rng(0), full_frame=True)
    assert sample.full_frame is True
    assert sample.origin is None
    assert sample.image.shape == (1, 20, 24)
    assert np.all(sample.mask == 0.0)
