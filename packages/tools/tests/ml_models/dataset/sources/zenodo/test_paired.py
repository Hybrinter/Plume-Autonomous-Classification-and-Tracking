"""Tests for the paired VIS+SWIR Zenodo source."""

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from tools.ml_models.dataset.sources.zenodo.bands import ZENODO_BAND_IDS
from tools.ml_models.dataset.sources.zenodo.paired import (
    PairedConfig,
    add_sensor_noise,
    camera_bins,
    gaussian_blur,
    load_paired_config,
    render_scene,
    resize_bilinear,
    scene_mix,
    sharpen_b1,
    swir_weights_for,
)
from tools.ml_models.dataset.sources.zenodo.prism import WeightTable

_IDS = list(ZENODO_BAND_IDS)
_TABLE = WeightTable(
    id="t",
    blue=MappingProxyType({"B1": 0.4, "B2": 0.6}),
    green=MappingProxyType({"B3": 1.0}),
    red=MappingProxyType({"B4": 1.0}),
)


def _flat(value_dn: float) -> np.ndarray:
    return np.full((len(_IDS), 120, 120), value_dn, dtype=np.float64)


def test_camera_bins_match_footprint_geometry() -> None:
    """el45 VIS and SWIR GSDs match the flight footprint values."""
    bins = camera_bins(PairedConfig())
    assert [item.spec.bin_id for item in bins] == ["el05", "el15", "el25", "el35", "el45"]
    el45 = bins[-1]
    assert el45.spec.lateral_m == pytest.approx(44.3, abs=0.2)
    assert el45.spec.along_m == pytest.approx(67.7, abs=0.2)
    assert el45.swir_gsd.lateral_m / el45.spec.lateral_m == pytest.approx(125 / 69, rel=1e-3)
    assert el45.spec.elevation_deg == 45.0


def test_flat_scene_gives_four_equal_flat_planes() -> None:
    """A flat reflectance stays flat in every plane with noise off."""
    for item in camera_bins(PairedConfig()):
        image = render_scene(
            _flat(2000.0), _IDS, _TABLE, PairedConfig(), item, {"B8A": 0.45, "B11": 0.55}
        )
        assert image.shape[0] == 4
        assert image.dtype == np.float32
        np.testing.assert_allclose(image, 0.2, atol=1e-5)


def test_swir_plane_keeps_less_contrast_than_vis() -> None:
    """A 120 m stripe pattern survives in VIS more than in SWIR."""
    stack = _flat(1000.0)
    stack[:, :, (np.arange(120) // 6) % 2 == 1] = 3000.0
    item = camera_bins(PairedConfig())[0]
    image = render_scene(stack, _IDS, _TABLE, PairedConfig(), item, {"B8A": 0.5, "B11": 0.5})
    assert image[3].std() < image[2].std()
    assert image[2].std() > 0.01


def test_centered_edge_stays_centered_in_both_planes() -> None:
    """A step at the window center is antisymmetric about the center in VIS and SWIR."""
    stack = _flat(1000.0)
    stack[:, :, 60:] = 5000.0
    cfg = PairedConfig(b1_sharpen=False)
    for item in camera_bins(cfg):
        image = render_scene(stack, _IDS, _TABLE, cfg, item, {"B8A": 0.5, "B11": 0.5})
        for plane in (image[2], image[3]):
            profile = plane.mean(axis=0)
            np.testing.assert_allclose(profile + profile[::-1], 0.6, atol=2e-3)


def test_resize_bilinear_is_identity_and_center_aligned() -> None:
    """Same size copies; a 2-px ramp upsampled to 4 keeps its midpoint."""
    planes = np.arange(12, dtype=np.float64).reshape(1, 3, 4)
    np.testing.assert_allclose(resize_bilinear(planes, (3, 4)), planes)
    up = resize_bilinear(np.asarray([[[0.0, 1.0]]]), (1, 4))[0, 0]
    np.testing.assert_allclose(up, [0.0, 0.25, 0.75, 1.0])
    with pytest.raises(ValueError):
        resize_bilinear(planes, (0, 4))


def test_gaussian_blur_preserves_mean_and_skips_tiny_sigma() -> None:
    """Blur keeps a flat plane flat and a tiny sigma is a no-op."""
    flat = np.full((1, 8, 8), 0.3)
    np.testing.assert_allclose(gaussian_blur(flat, (2.0, 2.0)), 0.3)
    rng = np.random.default_rng(0)
    noisy = rng.random((1, 8, 8))
    np.testing.assert_allclose(gaussian_blur(noisy, (0.0, 0.0)), noisy)
    assert gaussian_blur(noisy, (1.0, 1.0)).std() < noisy.std()


def test_sharpen_b1_injects_only_fine_b2_detail() -> None:
    """Flat B2 leaves B1 unchanged; fine B2 detail appears in B1."""
    stack = _flat(1000.0)
    np.testing.assert_allclose(sharpen_b1(stack, _IDS, 60.0)[0], 1000.0, atol=1e-6)
    stack[1, :, ::2] = 1400.0
    sharpened = sharpen_b1(stack, _IDS, 60.0)
    assert sharpened[0].std() > 100.0
    np.testing.assert_allclose(sharpened[1:], stack[1:])


def test_scene_mix_is_stable_and_seeded() -> None:
    """The same stem and seed give the same mix; the seed changes it."""
    assert scene_mix("1298_2019-10-25", 0) == scene_mix("1298_2019-10-25", 0)
    assert scene_mix("1298_2019-10-25", 0) != scene_mix("1298_2019-10-25", 1)
    assert 0.0 <= scene_mix("a", 0) < 1.0


def test_swir_weights_fixed_or_randomized_between_endpoints() -> None:
    """Fixed mode returns the config weights; random mode stays between endpoints."""
    fixed = PairedConfig()
    weights, mix = swir_weights_for(fixed, "s")
    assert mix is None
    assert dict(weights) == dict(fixed.swir_weights)
    rand = PairedConfig(swir_randomize=True)
    weights, mix = swir_weights_for(rand, "s")
    assert mix is not None
    assert sum(weights.values()) == pytest.approx(1.0)
    assert 0.3 - 1e-9 <= weights["B11"] <= 0.8 + 1e-9


def test_sensor_noise_matches_the_model() -> None:
    """Noise sigma at reflectance 0.5 is sqrt(e + r^2) / gain."""
    rng = np.random.default_rng(0)
    planes = np.full((1, 400, 400), 0.5)
    noisy = add_sensor_noise(planes, 10000.0, 20.0, 1.0, 16, rng)
    expected = np.sqrt(5000.0 + 20.0**2) / 10000.0
    assert noisy.std() == pytest.approx(expected, rel=0.03)
    assert noisy.mean() == pytest.approx(0.5, abs=1e-3)


def test_noise_requires_generator() -> None:
    """render_scene with noise and no generator raises."""
    item = camera_bins(PairedConfig())[0]
    with pytest.raises(ValueError, match="generator"):
        render_scene(_flat(2000.0), _IDS, _TABLE, PairedConfig(noise=True), item, {"B11": 1.0})


def test_render_rejects_missing_swir_band() -> None:
    """A SWIR weight on an absent band raises."""
    item = camera_bins(PairedConfig())[0]
    with pytest.raises(ValueError, match="missing band"):
        render_scene(_flat(2000.0), _IDS, _TABLE, PairedConfig(), item, {"B99": 1.0})


def test_config_validation_and_toml_loader(tmp_path: Path) -> None:
    """The loader applies overrides, normalizes weights, and rejects bad keys."""
    path = tmp_path / "paired.toml"
    path.write_text(
        'id = "x"\nelevations_deg = [10, 40]\nvis_f_number = 4\nnoise = true\n'
        "[swir_weights]\nB8A = 1\nB11 = 3\n",
        encoding="utf-8",
    )
    cfg = load_paired_config(path)
    assert cfg.id == "x"
    assert cfg.elevations_deg == (10.0, 40.0)
    assert cfg.vis_f_number == 4.0
    assert cfg.noise is True
    assert cfg.swir_weights["B11"] == pytest.approx(0.75)
    bad = tmp_path / "bad.toml"
    bad.write_text("bogus = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_paired_config(bad)
    with pytest.raises(ValueError):
        PairedConfig(elevations_deg=(5.0, 5.0))
    with pytest.raises(ValueError):
        PairedConfig(vis_f_number=0.0)


def test_scene_refs_carry_exact_gsd_elevation_and_mix() -> None:
    """Rows are measured GSD, carry the elevation, and tag the mix when randomized."""
    from tools.ml_models.dataset.sources.zenodo.archive import TileRef
    from tools.ml_models.dataset.sources.zenodo.paired import scene_refs

    tile = TileRef(
        stem="1298_2019-10-25T09:56:33.931Z_19",
        location_id="1298",
        positive=True,
        member_name="positive/x.tif",
        polygons=(),
    )
    cfg = PairedConfig(swir_randomize=True)
    refs = scene_refs(tile, cfg, camera_bins(cfg))
    assert [r.tile_id.rsplit("-", 1)[1] for r in refs] == ["el05", "el15", "el25", "el35", "el45"]
    assert not any(r.gsd_nominal for r in refs)
    assert refs[-1].theta_g_deg == 45.0
    assert {r.group_id for r in refs} == {"1298"}
    tags = {c.name: c.value for c in refs[0].metadata.conditions}
    assert tags["swir_mix"] == f"{scene_mix(tile.stem, 0):.2f}"
    fixed = scene_refs(tile, PairedConfig(), camera_bins(PairedConfig()))
    assert fixed[0].metadata.conditions == ()
