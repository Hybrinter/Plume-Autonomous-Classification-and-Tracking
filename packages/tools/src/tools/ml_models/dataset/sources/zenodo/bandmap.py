"""Sentinel-2 to camera band weights from modeled spectral responses.

Contains:
  - SpectralResponse: Protocol for a relative response at given wavelengths.
  - SuperGaussianBand: center/FWHM band shape (order 1 is a Gaussian).
  - LogisticPassband: smooth top-hat with half-power edges.
  - ProductResponse: product of factor responses (sensor x coating x lens).
  - S2A_BANDS, s2_band: Sentinel-2A band centers and FWHM as band shapes.
  - S2_NATIVE_GSD_M, NO_B1_BASIS: native band GSD and the B1-free VIS basis.
  - imx267_placeholder_channels, hd5_placeholder_channel: candidate cameras.
  - ChannelMapping, map_channel: weights and diagnostics for one channel.
  - vis_weight_table_toml: a VIS mapping as a prism weight-table file.
  - swir_edge_sensitivity: SWIR mappings over band-edge and basis cases.

A camera channel sees ``s = int(R E l rho) / int(R E l)`` where ``R`` is the
channel quantum efficiency, ``E`` is solar spectral irradiance (a 5772 K
blackbody, relative), ``l`` is wavelength (photon flux is ``E l / (h c)``, and
QE counts photons), and ``rho`` is surface reflectance. Reflectance between Sentinel-2 band
centers is linear in wavelength (tent basis on the sorted basis-band
centers, constant beyond the end centers), so the weight of basis band ``b``
is ``int(R E l h_b) / int(R E l)``. The tents are a partition of unity, so the
weights of one channel sum to 1 and a flat reflectance maps to itself.

The weights alone hide throughput. ``band_capture`` reports, for each
Sentinel-2 band, the fraction of that band's solar-weighted response that the
channel passes. ``coverage`` reports the fraction of the channel signal that
falls inside any basis band; the rest is interpolated across a spectral gap.

B10 is L1C top-of-atmosphere cirrus reflectance and the other corpus bands
are L2A surface reflectance, so B10 is never a default basis band. B8
overlaps B8A and B9 is a water-vapour band; both are left out of the default
basis. B1 is a 60 m coastal-aerosol band; ``NO_B1_BASIS`` drops it so blue
extends B2 flat below 492 nm. Camera curves are placeholders until datasheet curves exist; their
function names carry ``placeholder``. ``vis_weight_table_toml`` stores the caller's table id and
rejects only an empty one; the CLI default id ``imx267_placeholder_v0`` carries ``placeholder``.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np

_H = 6.62607015e-34  # Planck constant, J s
_C = 2.99792458e8  # speed of light, m/s
_K_B = 1.380649e-23  # Boltzmann constant, J/K
SUN_TEMPERATURE_K = 5772.0
GRID_LOW_M = 350e-9
GRID_HIGH_M = 2500e-9
GRID_STEP_M = 0.5e-9

S2A_BANDS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "B1": (442.7e-9, 21e-9),
        "B2": (492.4e-9, 66e-9),
        "B3": (559.8e-9, 36e-9),
        "B4": (664.6e-9, 31e-9),
        "B5": (704.1e-9, 15e-9),
        "B6": (740.5e-9, 15e-9),
        "B7": (782.8e-9, 20e-9),
        "B8": (832.8e-9, 106e-9),
        "B8A": (864.7e-9, 21e-9),
        "B9": (945.1e-9, 20e-9),
        "B10": (1373.5e-9, 31e-9),
        "B11": (1613.7e-9, 91e-9),
        "B12": (2202.4e-9, 175e-9),
    }
)
DEFAULT_BASIS: tuple[str, ...] = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8A",
    "B11",
    "B12",
)
S2_NATIVE_GSD_M: Mapping[str, float] = MappingProxyType(
    {
        "B1": 60.0,
        "B2": 10.0,
        "B3": 10.0,
        "B4": 10.0,
        "B5": 20.0,
        "B6": 20.0,
        "B7": 20.0,
        "B8": 10.0,
        "B8A": 20.0,
        "B9": 60.0,
        "B10": 60.0,
        "B11": 20.0,
        "B12": 20.0,
    }
)
NO_B1_BASIS: tuple[str, ...] = tuple(band_id for band_id in DEFAULT_BASIS if band_id != "B1")
_S2_BAND_ORDER = 3
_COLORS: tuple[str, ...] = ("blue", "green", "red")


@runtime_checkable
class SpectralResponse(Protocol):
    """Relative spectral response at given wavelengths."""

    def response(self, wavelength_m: np.ndarray) -> np.ndarray:
        """Return the response at each wavelength.

        Args:
            wavelength_m: Wavelengths in metres. # np.ndarray[float64, (N,)]

        Returns:
            np.ndarray[float64, (N,)]: Non-negative relative response.
        """
        ...


@dataclass(frozen=True, slots=True)
class SuperGaussianBand:
    """Band shape ``peak * exp(-ln2 * |2 (l - c) / fwhm| ** (2 order))``.

    Attributes:
        center_m: Center wavelength in metres.
        fwhm_m: Full width at half maximum in metres.
        order: 1 is a Gaussian; larger orders flatten the top and sharpen
            the edges. The half-maximum points stay at ``c +/- fwhm / 2``.
        peak: Response at the center.
    """

    center_m: float
    fwhm_m: float
    order: int = 1
    peak: float = 1.0

    def response(self, wavelength_m: np.ndarray) -> np.ndarray:
        """Return the band response.

        Args:
            wavelength_m: Wavelengths in metres. # np.ndarray[float64, (N,)]

        Returns:
            np.ndarray[float64, (N,)]: Response in ``[0, peak]``.

        Raises:
            ValueError: If the FWHM, order, or peak is not positive.
        """
        if self.fwhm_m <= 0 or self.order < 1 or self.peak <= 0:
            raise ValueError("band FWHM, order, and peak must be positive")
        x = np.abs(2.0 * (np.asarray(wavelength_m, dtype=np.float64) - self.center_m))
        return np.asarray(
            self.peak * np.exp(-math.log(2.0) * (x / self.fwhm_m) ** (2 * self.order)),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class LogisticPassband:
    """Smooth top-hat ``sigma((l - low) / w) * sigma((high - l) / w)``.

    Attributes:
        low_m: Lower half-power wavelength in metres.
        high_m: Upper half-power wavelength in metres.
        edge_width_m: Logistic scale ``w`` in metres; the 10-90 percent rise
            spans about ``4.4 w``.

    Notes:
        A coating or detector cut-on/cut-off with no published curve is
        modeled this way. The half-power points sit at ``low`` and ``high``
        when the band is much wider than ``w``.
    """

    low_m: float
    high_m: float
    edge_width_m: float

    def response(self, wavelength_m: np.ndarray) -> np.ndarray:
        """Return the passband transmission.

        Args:
            wavelength_m: Wavelengths in metres. # np.ndarray[float64, (N,)]

        Returns:
            np.ndarray[float64, (N,)]: Transmission in ``[0, 1]``.

        Raises:
            ValueError: If ``high <= low`` or the edge width is not positive.
        """
        if self.high_m <= self.low_m or self.edge_width_m <= 0:
            raise ValueError("passband needs high > low and a positive edge width")
        wl = np.asarray(wavelength_m, dtype=np.float64)
        rise = 0.5 * (1.0 + np.tanh((wl - self.low_m) / (2.0 * self.edge_width_m)))
        fall = 0.5 * (1.0 + np.tanh((self.high_m - wl) / (2.0 * self.edge_width_m)))
        return np.asarray(rise * fall, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ProductResponse:
    """Product of factor responses, such as sensor QE x coating x lens.

    Attributes:
        factors: Responses multiplied pointwise. At least one is required.
    """

    factors: tuple[SpectralResponse, ...]

    def response(self, wavelength_m: np.ndarray) -> np.ndarray:
        """Return the product of all factor responses.

        Args:
            wavelength_m: Wavelengths in metres. # np.ndarray[float64, (N,)]

        Returns:
            np.ndarray[float64, (N,)]: Pointwise product.

        Raises:
            ValueError: If there are no factors.
        """
        if not self.factors:
            raise ValueError("a product response needs at least one factor")
        out = np.ones_like(np.asarray(wavelength_m, dtype=np.float64))
        for factor in self.factors:
            out = out * factor.response(wavelength_m)
        return out


@dataclass(frozen=True, slots=True)
class ChannelMapping:
    """Sentinel-2 weights and diagnostics for one camera channel.

    Attributes:
        name: Channel name.
        weights: Basis band id to weight; the weights sum to 1.
        band_capture: Every Sentinel-2 band id to the fraction of that band's
            solar-weighted response the channel passes, in ``[0, 1]``.
        coverage: Fraction of the channel signal inside any basis band,
            taking the pointwise maximum of the basis band shapes.
        centroid_m: Solar-weighted response centroid in metres.
    """

    name: str
    weights: Mapping[str, float]
    band_capture: Mapping[str, float]
    coverage: float
    centroid_m: float


def wavelength_grid() -> np.ndarray:
    """Return the uniform integration grid.

    Returns:
        np.ndarray[float64, (N,)]: ``GRID_LOW_M`` to ``GRID_HIGH_M`` in steps of
        ``GRID_STEP_M``, metres.
    """
    count = round((GRID_HIGH_M - GRID_LOW_M) / GRID_STEP_M) + 1
    return np.linspace(GRID_LOW_M, GRID_HIGH_M, count, dtype=np.float64)


def solar_blackbody(
    wavelength_m: np.ndarray, temperature_k: float = SUN_TEMPERATURE_K
) -> np.ndarray:
    """Return Planck spectral radiance normalized to its peak on the input grid.

    Args:
        wavelength_m: Wavelengths in metres. # np.ndarray[float64, (N,)]
        temperature_k: Blackbody temperature in kelvin.

    Returns:
        np.ndarray[float64, (N,)]: Relative irradiance with maximum 1.

    Notes:
        Only the spectral shape matters because every weight is a ratio.
        Atmospheric absorption is not modeled.
    """
    wl = np.asarray(wavelength_m, dtype=np.float64)
    radiance = (2.0 * _H * _C**2 / wl**5) / np.expm1(_H * _C / (wl * _K_B * temperature_k))
    return np.asarray(radiance / np.max(radiance), dtype=np.float64)


def s2_band(band_id: str) -> SuperGaussianBand:
    """Return the Sentinel-2A band shape for ``band_id``.

    Args:
        band_id: Canonical id such as ``B8A``.

    Returns:
        SuperGaussianBand: Order-3 super-Gaussian at the published center and
        FWHM. Published response functions are flat-topped with steep edges.

    Raises:
        KeyError: If the id is not a Sentinel-2 band.
    """
    center, fwhm = S2A_BANDS[band_id]
    return SuperGaussianBand(center_m=center, fwhm_m=fwhm, order=_S2_BAND_ORDER)


def imx267_placeholder_channels(
    coating_low_m: float = 425e-9,
    coating_high_m: float = 675e-9,
    coating_edge_m: float = 5e-9,
) -> dict[str, SpectralResponse]:
    """Return placeholder IMX267 blue, green, red responses behind the VIS coating.

    Args:
        coating_low_m: Coating lower half-power wavelength in metres.
        coating_high_m: Coating upper half-power wavelength in metres.
        coating_edge_m: Coating logistic edge scale in metres.

    Returns:
        dict[str, SpectralResponse]: ``blue``, ``green``, ``red`` responses.

    Notes:
        The color filters are Gaussians (blue 465 nm / 100 nm FWHM, green
        535 / 100, red 610 / 90). These are placeholders, not datasheet
        values; replace them with the measured IMX267 color QE. Lens
        transmission is taken as flat across the coating band.
    """
    coating = LogisticPassband(coating_low_m, coating_high_m, coating_edge_m)
    filters = {
        "blue": SuperGaussianBand(center_m=465e-9, fwhm_m=100e-9),
        "green": SuperGaussianBand(center_m=535e-9, fwhm_m=100e-9),
        "red": SuperGaussianBand(center_m=610e-9, fwhm_m=90e-9),
    }
    return {name: ProductResponse((band, coating)) for name, band in filters.items()}


def hd5_placeholder_channel(
    low_m: float = 1000e-9,
    high_m: float = 1650e-9,
    edge_m: float = 10e-9,
) -> SpectralResponse:
    """Return a placeholder Attollo HD5 InGaAs response.

    Args:
        low_m: Cut-on half-power wavelength in metres.
        high_m: Cut-off half-power wavelength in metres.
        edge_m: Logistic edge scale in metres.

    Returns:
        SpectralResponse: Flat-topped passband.

    Notes:
        The datasheet states 1.0-1.65 um and the product page 0.9-1.7 um;
        both are run as sensitivity cases. The response inside the band is
        flat until the vendor QE curve is available.
    """
    return LogisticPassband(low_m, high_m, edge_m)


def map_channel(
    name: str,
    channel: SpectralResponse,
    basis: Sequence[str] = DEFAULT_BASIS,
    solar_weighted: bool = True,
    photon_counting: bool = True,
) -> ChannelMapping:
    """Compute Sentinel-2 weights and diagnostics for one camera channel.

    Args:
        name: Channel name copied to the result.
        channel: Camera channel response.
        basis: Sentinel-2 band ids used as interpolation nodes. Ids are sorted
            by center wavelength.
        solar_weighted: Multiply the channel response by the solar blackbody.
        photon_counting: Multiply by wavelength, so a QE response integrates
            photon flux rather than energy.

    Returns:
        ChannelMapping: Weights over ``basis`` and diagnostics over all bands.

    Raises:
        ValueError: If the basis is empty, repeats an id, or names an unknown
            band, or if the channel response integrates to 0 on the grid.
    """
    if not basis or len(set(basis)) != len(basis):
        raise ValueError("basis must be a non-empty set of band ids")
    unknown = sorted(set(basis) - set(S2A_BANDS))
    if unknown:
        raise ValueError(f"unknown basis bands {unknown}")
    wl = wavelength_grid()
    sun = solar_blackbody(wl) if solar_weighted else np.ones_like(wl)
    if photon_counting:
        sun = sun * wl / GRID_HIGH_M
    signal = channel.response(wl) * sun  # np.ndarray[float64, (N,)]
    total = float(np.trapezoid(signal, wl))
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"channel {name!r} has no response on the grid")
    ordered = sorted(basis, key=lambda band_id: S2A_BANDS[band_id][0])
    centers = np.asarray([S2A_BANDS[band_id][0] for band_id in ordered], dtype=np.float64)
    weights: dict[str, float] = {}
    for index, band_id in enumerate(ordered):
        hat = np.zeros(len(ordered), dtype=np.float64)
        hat[index] = 1.0
        tent = np.interp(wl, centers, hat)  # constant beyond the end centers
        weights[band_id] = float(np.trapezoid(signal * tent, wl)) / total
    capture: dict[str, float] = {}
    for band_id in S2A_BANDS:
        shape = s2_band(band_id).response(wl) * sun
        capture[band_id] = float(np.trapezoid(shape * channel.response(wl), wl)) / float(
            np.trapezoid(shape, wl)
        )
    envelope = np.max(np.stack([s2_band(band_id).response(wl) for band_id in ordered]), axis=0)
    coverage = float(np.trapezoid(signal * envelope, wl)) / total
    centroid = float(np.trapezoid(signal * wl, wl)) / total
    return ChannelMapping(
        name=name,
        weights=MappingProxyType(weights),
        band_capture=MappingProxyType(capture),
        coverage=coverage,
        centroid_m=centroid,
    )


def vis_weight_table_toml(
    table_id: str,
    mappings: Mapping[str, ChannelMapping],
    digits: int = 6,
    floor: float = 1e-4,
) -> str:
    """Render blue, green, red mappings as a prism weight-table TOML file.

    Args:
        table_id: Table identifier; it is stored on every dataset manifest.
        mappings: ``blue``, ``green``, ``red`` channel mappings.
        digits: Decimal places written per weight.
        floor: Weights below this are dropped before renormalizing.

    Returns:
        str: TOML text that ``prism.load_weight_table`` accepts.

    Raises:
        ValueError: If a color is missing or the id is empty.

    Notes:
        After rounding, the residual goes to the largest weight so each color
        sums to 1 within the loader's 1e-6 tolerance.
    """
    if not table_id:
        raise ValueError("table id must be non-empty")
    missing = [color for color in _COLORS if color not in mappings]
    if missing:
        raise ValueError(f"missing colors {missing}")
    lines = [
        "# Generated by tools.ml_models.dataset.sources.zenodo.bandmap.",
        "# Camera curves are placeholders until datasheet curves exist.",
        f'id = "{table_id}"',
    ]
    for color in _COLORS:
        mapping = mappings[color]
        kept = {k: v for k, v in mapping.weights.items() if v >= floor}
        scale = sum(kept.values())
        rounded = {k: round(v / scale, digits) for k, v in kept.items()}
        largest = max(rounded, key=lambda k: rounded[k])
        rounded[largest] = round(rounded[largest] + 1.0 - sum(rounded.values()), digits)
        lines.append("")
        lines.append(
            f"# coverage {mapping.coverage:.3f}, centroid {mapping.centroid_m * 1e9:.1f} nm"
        )
        lines.append(f"[{color}]")
        lines.extend(f"{k} = {v:.{digits}f}" for k, v in rounded.items())
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class SensitivityCase:
    """One SWIR band-edge and basis case.

    Attributes:
        label: Case name.
        low_m: Cut-on wavelength in metres.
        high_m: Cut-off wavelength in metres.
        basis: Basis band ids.
        mapping: Resulting channel mapping.
    """

    label: str
    low_m: float
    high_m: float
    basis: tuple[str, ...]
    mapping: ChannelMapping


def swir_edge_sensitivity(edge_m: float = 10e-9) -> tuple[SensitivityCase, ...]:
    """Map the HD5 channel over its two published bands and two basis sets.

    Args:
        edge_m: Logistic edge scale in metres.

    Returns:
        tuple[SensitivityCase, ...]: Datasheet (1.0-1.65 um) and product-page
        (0.9-1.7 um) edges, each with the default basis and with B9 added.

    Notes:
        B9 is an absorption band, so adding it as a node shows how much the
        weights depend on a band whose L2A value is mostly atmosphere.
    """
    with_b9 = tuple(sorted((*DEFAULT_BASIS, "B9"), key=lambda band_id: S2A_BANDS[band_id][0]))
    cases: list[SensitivityCase] = []
    for label, low, high in (("datasheet", 1000e-9, 1650e-9), ("product_page", 900e-9, 1700e-9)):
        for basis_label, basis in (("default", DEFAULT_BASIS), ("with_b9", with_b9)):
            channel = hd5_placeholder_channel(low, high, edge_m)
            cases.append(
                SensitivityCase(
                    label=f"{label}/{basis_label}",
                    low_m=low,
                    high_m=high,
                    basis=tuple(basis),
                    mapping=map_channel("swir", channel, basis),
                )
            )
    return tuple(cases)
