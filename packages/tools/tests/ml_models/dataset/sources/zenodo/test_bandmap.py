"""Tests for Sentinel-2 to camera band weights."""

from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.dataset.sources.zenodo.bandmap import (
    DEFAULT_BASIS,
    NO_B1_BASIS,
    S2_NATIVE_GSD_M,
    S2A_BANDS,
    LogisticPassband,
    ProductResponse,
    SpectralResponse,
    SuperGaussianBand,
    hd5_placeholder_channel,
    imx267_placeholder_channels,
    map_channel,
    swir_edge_sensitivity,
    vis_weight_table_toml,
    wavelength_grid,
)
from tools.ml_models.dataset.sources.zenodo.prism import load_weight_table


def test_band_shapes_have_half_power_at_their_edges() -> None:
    """Super-Gaussian FWHM and logistic edges sit at half response."""
    band = SuperGaussianBand(center_m=500e-9, fwhm_m=40e-9, order=3)
    edges = band.response(np.asarray([480e-9, 500e-9, 520e-9]))
    np.testing.assert_allclose(edges, [0.5, 1.0, 0.5], atol=1e-12)
    passband = LogisticPassband(1000e-9, 1650e-9, 10e-9)
    np.testing.assert_allclose(
        passband.response(np.asarray([1000e-9, 1650e-9])), [0.5, 0.5], atol=1e-9
    )


def test_responses_satisfy_the_protocol() -> None:
    """Every response type is a SpectralResponse."""
    for item in (
        SuperGaussianBand(500e-9, 40e-9),
        LogisticPassband(400e-9, 700e-9, 5e-9),
        ProductResponse((SuperGaussianBand(500e-9, 40e-9),)),
    ):
        assert isinstance(item, SpectralResponse)


def test_invalid_shapes_are_rejected() -> None:
    """Non-positive widths, inverted passbands, and empty products raise."""
    wl = wavelength_grid()
    with pytest.raises(ValueError):
        SuperGaussianBand(500e-9, 0.0).response(wl)
    with pytest.raises(ValueError):
        LogisticPassband(700e-9, 400e-9, 5e-9).response(wl)
    with pytest.raises(ValueError):
        ProductResponse(()).response(wl)


def test_narrow_channel_on_a_basis_center_maps_to_that_band() -> None:
    """A narrow channel at the B3 center takes nearly all weight from B3."""
    center = S2A_BANDS["B3"][0]
    mapping = map_channel("narrow", SuperGaussianBand(center, 2e-9))
    assert mapping.weights["B3"] > 0.99
    assert mapping.coverage > 0.99


def test_channel_between_two_centers_splits_by_distance() -> None:
    """A narrow channel midway between B8A and B11 splits about evenly."""
    mid = 0.5 * (S2A_BANDS["B8A"][0] + S2A_BANDS["B11"][0])
    mapping = map_channel("gap", SuperGaussianBand(mid, 2e-9), solar_weighted=False)
    assert mapping.weights["B8A"] == pytest.approx(0.5, abs=1e-3)
    assert mapping.weights["B11"] == pytest.approx(0.5, abs=1e-3)
    assert mapping.coverage < 1e-3


def test_weights_sum_to_one_for_every_channel() -> None:
    """Tent weights form a partition of unity."""
    channels = dict(imx267_placeholder_channels())
    channels["swir"] = hd5_placeholder_channel()
    for name, channel in channels.items():
        mapping = map_channel(name, channel)
        assert set(mapping.weights) == set(DEFAULT_BASIS)
        assert sum(mapping.weights.values()) == pytest.approx(1.0, abs=1e-9)


def test_vis_coating_reduces_b4_capture() -> None:
    """The 675 nm coating edge passes less of B4 than an open red filter."""
    red = SuperGaussianBand(center_m=610e-9, fwhm_m=90e-9)
    coated = imx267_placeholder_channels()["red"]
    assert (
        map_channel("red", coated).band_capture["B4"] < map_channel("red", red).band_capture["B4"]
    )


def test_swir_lower_edge_controls_b9() -> None:
    """B9 is outside the 1.0 um band and inside the 0.9 um band."""
    datasheet = map_channel("swir", hd5_placeholder_channel(1000e-9, 1650e-9))
    product = map_channel("swir", hd5_placeholder_channel(900e-9, 1700e-9))
    assert datasheet.band_capture["B9"] < 0.05
    assert product.band_capture["B9"] > 0.9
    assert datasheet.band_capture["B11"] < product.band_capture["B11"]


def test_b10_is_not_a_default_basis_band() -> None:
    """The L1C cirrus band never enters the default weights."""
    assert "B10" not in DEFAULT_BASIS
    mapping = map_channel("swir", hd5_placeholder_channel())
    assert "B10" not in mapping.weights


def test_bad_basis_is_rejected() -> None:
    """Empty, repeated, or unknown basis ids raise."""
    channel = hd5_placeholder_channel()
    for basis in ((), ("B2", "B2"), ("B2", "B99")):
        with pytest.raises(ValueError):
            map_channel("swir", channel, basis)


def test_zero_response_is_rejected() -> None:
    """A channel with no response on the grid raises."""
    with pytest.raises(ValueError, match="no response"):
        map_channel("dark", SuperGaussianBand(10e-6, 1e-9))


def test_vis_table_round_trips_through_the_prism_loader(tmp_path: Path) -> None:
    """The generated TOML loads and each color sums to 1."""
    mappings = {name: map_channel(name, ch) for name, ch in imx267_placeholder_channels().items()}
    path = tmp_path / "vis.toml"
    path.write_text(vis_weight_table_toml("imx267_placeholder_v0", mappings), encoding="utf-8")
    table = load_weight_table(path)
    assert table.id == "imx267_placeholder_v0"
    for color in ("blue", "green", "red"):
        weights = table.color_map(color)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert "B10" not in weights
    assert max(table.color_map("blue"), key=lambda k: table.color_map("blue")[k]) in {"B1", "B2"}


def test_vis_table_requires_all_colors() -> None:
    """A missing color or empty id raises."""
    mapping = map_channel("blue", imx267_placeholder_channels()["blue"])
    with pytest.raises(ValueError, match="missing"):
        vis_weight_table_toml("t", {"blue": mapping})
    with pytest.raises(ValueError, match="id"):
        vis_weight_table_toml("", {"blue": mapping})


def test_swir_sensitivity_covers_both_edges_and_bases() -> None:
    """Four cases; adding B9 moves weight onto B9."""
    cases = {case.label: case for case in swir_edge_sensitivity()}
    assert set(cases) == {
        "datasheet/default",
        "datasheet/with_b9",
        "product_page/default",
        "product_page/with_b9",
    }
    assert "B9" not in cases["datasheet/default"].mapping.weights
    assert cases["product_page/with_b9"].mapping.weights["B9"] > 0.1


def test_photon_counting_shifts_swir_weight_toward_b11() -> None:
    """Photon weighting favors long wavelengths across a wide passband."""
    channel = hd5_placeholder_channel(900e-9, 1700e-9)
    photons = map_channel("swir", channel, photon_counting=True)
    energy = map_channel("swir", channel, photon_counting=False)
    assert photons.weights["B11"] > energy.weights["B11"]
    assert photons.centroid_m > energy.centroid_m


def test_no_b1_basis_moves_blue_weight_to_b2() -> None:
    """Without B1, blue draws on B2 and every basis band has a native GSD."""
    blue = imx267_placeholder_channels()["blue"]
    without = map_channel("blue", blue, NO_B1_BASIS)
    assert "B1" not in without.weights
    assert without.weights["B2"] > map_channel("blue", blue).weights["B2"]
    assert set(S2_NATIVE_GSD_M) == set(S2A_BANDS)
