"""Paired VIS+SWIR Zenodo source for the candidate two-camera payload.

Contains:
  - PairedConfig, load_paired_config: camera, blur, noise, and SWIR-mix settings.
  - camera_bins: VIS GSD bins and matching SWIR GSD from flight footprint geometry.
  - sharpen_b1: inject B2 detail finer than 60 m into the smooth B1 plane.
  - gaussian_blur, resize_bilinear: separable blur and center-aligned resize.
  - add_sensor_noise: shot, read, and quantization noise in reflectance units.
  - render_scene: one Zenodo stack to BLUE, GREEN, RED, SWIR planes.
  - scene_refs: one scene's rows across camera bins.
  - ZenodoPairedSource: RawSource emitting every scene at every camera bin.

Every row is stored on the VIS band-sample grid (6.9 um pitch, 2 x the IMX267
mosaic pitch), so ``gsd`` is metres per VIS band sample. The SWIR plane is
blurred and area-sampled at the SWIR GSD, then resized onto the VIS grid with
pixel centers aligned over the same 1200 m window; early fusion on flight
resamples the same way. Both GSDs come from ``flight.payload.gimbal.footprint``
at the frame center for a nominal circular orbit. Rows are not
``gsd_nominal``: the stored image is resampled to exactly that GSD, as in
``ZenodoSource``; only the elevation is notional.

Optics blur is a Gaussian fit to the Airy core, ``sigma = 0.42 lambda N`` at
the focal plane, applied on the native 10 m grid before area sampling; area
sampling supplies the pixel aperture. Noise is off by default. The SWIR mix
is fixed by default; with ``swir_randomize`` each scene draws a mix between
two endpoint weight maps from a CRC32 seed of its stem, so every bin of a
scene shares one mix and rebuilds are identical. The mix is recorded on each
row as condition ``swir_mix``.

Camera values are estimates, not datasheet curves or measurements.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math
import tomllib
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
from flight.payload.gimbal.footprint import nominal_iss_state, pixel_gsd
from flight.payload.gimbal.intersect import CameraGeometry

from tools.ml_models.dataset.raw import (
    BinSpec,
    ConditionTag,
    GsdPair,
    RawTile,
    RawTileRef,
)
from tools.ml_models.dataset.sources.zenodo.annotations import rasterize_percent_mask
from tools.ml_models.dataset.sources.zenodo.archive import (
    TileRef,
    build_index,
    iter_stacks,
    observation_metadata,
    to_native_stack,
)
from tools.ml_models.dataset.sources.zenodo.bands import coerce_descriptions, verify_band_order
from tools.ml_models.dataset.sources.zenodo.bins import EXTENT_M, NATIVE_SIDE, actual_gsd, bin_hw
from tools.ml_models.dataset.sources.zenodo.prism import WeightTable, mix_prism
from tools.ml_models.dataset.sources.zenodo.resample import resample_area

PAIRED_BANDS: tuple[str, ...] = ("BLUE", "GREEN", "RED", "SWIR")
NATIVE_GSD_M = EXTENT_M / NATIVE_SIDE
_DN_SCALE = 10000.0
_AIRY_SIGMA = 0.42  # Gaussian sigma of the Airy core per (lambda N)
_WGS84_A_M = 6_378_137.0
_WGS84_F = 1.0 / 298.257223563
_MU_M3_S2 = 3.986004418e14
_OMEGA_EARTH_RAD_S = 7.2921159e-5


def _default_swir_weights() -> Mapping[str, float]:
    """Return the HD5 datasheet-edge weights from the band mapper."""
    from tools.ml_models.dataset.sources.zenodo.bandmap import (
        hd5_placeholder_channel,
        map_channel,
    )

    return _normalized(map_channel("swir", hd5_placeholder_channel()).weights)


def _normalized(weights: Mapping[str, float], floor: float = 1e-4) -> Mapping[str, float]:
    """Drop weights below ``floor`` and rescale the rest to sum to 1."""
    kept = {k: float(v) for k, v in weights.items() if v >= floor}
    total = sum(kept.values())
    if not kept or not math.isfinite(total) or total <= 0:
        raise ValueError("weights must keep a positive finite total")
    return MappingProxyType({k: v / total for k, v in kept.items()})


def _is_normalized(weights: Mapping[str, float]) -> bool:
    """Return whether ``weights`` already equals its :func:`_normalized` form."""
    target = _normalized(weights)
    return set(target) == set(weights) and all(
        math.isclose(float(weights[k]), v, rel_tol=1e-9, abs_tol=1e-12) for k, v in target.items()
    )


@dataclass(frozen=True, slots=True)
class PairedConfig:
    """Settings for the paired VIS+SWIR build.

    Attributes:
        id: Config identifier, joined to the VIS table id on the manifest.
        bands: Output planes, a subset of ``PAIRED_BANDS`` in that order.
            VIS-only and SWIR-only builds share splits and rendering with
            the paired build.
        elevations_deg: Gimbal elevations; one bin per value.
        altitude_m: Nominal orbit altitude.
        plume_height_m: Height above the ellipsoid where GSD is evaluated.
        vis_width_px: VIS band-plane width (3840 px ROI / 2).
        vis_height_px: VIS band-plane height.
        vis_pitch_m: VIS band-sample pitch.
        vis_focal_m: VIS focal length.
        vis_f_number: VIS working f-number (sheet f/11; 16-849 is f/4).
        vis_wavelength_m: VIS wavelength used for the diffraction blur.
        swir_width_px: SWIR width.
        swir_height_px: SWIR height.
        swir_pitch_m: SWIR pixel pitch.
        swir_focal_m: SWIR focal length.
        swir_f_number: SWIR f-number.
        swir_wavelength_m: SWIR wavelength used for the diffraction blur.
        extra_blur_px: Added Gaussian sigma in output pixels (jitter, defocus).
        b1_sharpen: Inject B2 detail into B1 before mixing VIS.
        b1_native_m: B1 native resolution.
        swir_weights: Fixed Sentinel-2 weights for the SWIR plane.
        swir_randomize: Draw a per-scene mix between the two endpoints.
        swir_weights_low: Endpoint at mix 0.
        swir_weights_high: Endpoint at mix 1.
        noise: Add shot, read, and quantization noise.
        saturation_reflectance: Reflectance that fills the full well.
        vis_full_well_e: VIS full-well capacity in electrons (estimate).
        vis_read_noise_e: VIS read noise in electrons (estimate).
        swir_full_well_e: SWIR full-well capacity in electrons (estimate).
        swir_read_noise_e: SWIR read noise in electrons (estimate).
        bit_depth: ADC bits for both cameras.
        seed: Base seed for mix and noise draws.
    """

    id: str = "vis_swir_paired_estimate_v1"
    bands: tuple[str, ...] = PAIRED_BANDS
    elevations_deg: tuple[float, ...] = (5.0, 15.0, 25.0, 35.0, 45.0)
    altitude_m: float = 440e3
    plume_height_m: float = 2000.0
    vis_width_px: int = 1920
    vis_height_px: int = 1080
    vis_pitch_m: float = 6.9e-6
    vis_focal_m: float = 0.1
    vis_f_number: float = 11.0
    vis_wavelength_m: float = 550e-9
    swir_width_px: int = 1280
    swir_height_px: int = 1024
    swir_pitch_m: float = 5e-6
    swir_focal_m: float = 0.05
    swir_f_number: float = 2.8
    swir_wavelength_m: float = 1300e-9
    extra_blur_px: float = 0.0
    b1_sharpen: bool = True
    b1_native_m: float = 60.0
    swir_weights: Mapping[str, float] = field(default_factory=_default_swir_weights)
    swir_randomize: bool = False
    swir_weights_low: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({"B8A": 0.7, "B11": 0.3})
    )
    swir_weights_high: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({"B8A": 0.2, "B11": 0.8})
    )
    noise: bool = False
    saturation_reflectance: float = 1.0
    vis_full_well_e: float = 10500.0
    vis_read_noise_e: float = 2.5
    swir_full_well_e: float = 30000.0
    swir_read_noise_e: float = 40.0
    bit_depth: int = 12
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject values that make the geometry, blur, or noise undefined."""
        positive = (
            self.altitude_m,
            self.vis_pitch_m,
            self.vis_focal_m,
            self.vis_f_number,
            self.vis_wavelength_m,
            self.swir_pitch_m,
            self.swir_focal_m,
            self.swir_f_number,
            self.swir_wavelength_m,
            self.b1_native_m,
            self.saturation_reflectance,
            self.vis_full_well_e,
            self.swir_full_well_e,
        )
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("geometry, optics, and sensor values must be positive")
        if not self.bands or self.bands != tuple(b for b in PAIRED_BANDS if b in self.bands):
            raise ValueError(f"bands must be a non-empty ordered subset of {PAIRED_BANDS}")
        if not self.id or not self.elevations_deg:
            raise ValueError("config id and elevations must be non-empty")
        if len(set(self.elevations_deg)) != len(self.elevations_deg):
            raise ValueError("elevations must be unique")
        if min(self.vis_width_px, self.vis_height_px, self.swir_width_px, self.swir_height_px) < 1:
            raise ValueError("sensor sizes must be positive")
        if self.extra_blur_px < 0 or self.vis_read_noise_e < 0 or self.swir_read_noise_e < 0:
            raise ValueError("blur and read noise must be non-negative")
        if not 1 <= self.bit_depth <= 16:
            raise ValueError("bit_depth must be in 1..16")
        for weights in (self.swir_weights, self.swir_weights_low, self.swir_weights_high):
            if not _is_normalized(weights):
                raise ValueError("SWIR weight maps must be normalized; load_paired_config does it")


def load_paired_config(path: str | Path) -> PairedConfig:
    """Load a PairedConfig from TOML; missing keys keep the defaults.

    Args:
        path: TOML file with PairedConfig field names. Weight maps are tables.

    Returns:
        PairedConfig: Parsed settings.

    Raises:
        OSError / tomllib.TOMLDecodeError: On a missing or malformed file.
        ValueError: On an unknown key or a value of the wrong type.
    """
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    known = {item.name: item for item in fields(PairedConfig)}
    extra = sorted(set(raw) - set(known))
    if extra:
        raise ValueError(f"unknown paired config keys {extra}")
    base = PairedConfig()
    values: dict[str, object] = {}
    for key, value in raw.items():
        current = getattr(base, key)
        if isinstance(current, Mapping):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a table of band weights")
            values[key] = _normalized({str(k): float(v) for k, v in value.items()})
        elif isinstance(current, tuple):
            if not isinstance(value, list):
                raise ValueError(f"{key} must be an array")
            if current and isinstance(current[0], str):
                values[key] = tuple(str(v) for v in value)
            else:
                values[key] = tuple(float(v) for v in value)
        elif isinstance(current, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
            values[key] = value
        elif isinstance(current, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
            values[key] = value
        elif isinstance(current, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number")
            values[key] = float(value)
        else:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            values[key] = value
    return replace(base, **values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class CameraBin:
    """One elevation bin with both camera GSDs.

    Attributes:
        spec: VIS bin on the manifest (VIS GSD, elevation).
        swir_gsd: SWIR metres per pixel at the same elevation.
    """

    spec: BinSpec
    swir_gsd: GsdPair


def camera_bins(cfg: PairedConfig) -> tuple[CameraBin, ...]:
    """Return one bin per elevation from flight footprint geometry.

    Args:
        cfg: Paired settings.

    Returns:
        tuple[CameraBin, ...]: Bins named ``elNN`` in ``cfg`` order.

    Raises:
        ValueError: If an elevation misses the Earth or a GSD is below the
            native 10 m Zenodo grid.
    """
    state = nominal_iss_state(cfg.altitude_m, _WGS84_A_M, _MU_M3_S2)
    if state is None:
        raise ValueError("invalid nominal orbit")
    r, v, utc = state
    vis = CameraGeometry(cfg.vis_width_px, cfg.vis_height_px, cfg.vis_pitch_m, cfg.vis_focal_m)
    swir = CameraGeometry(cfg.swir_width_px, cfg.swir_height_px, cfg.swir_pitch_m, cfg.swir_focal_m)
    out: list[CameraBin] = []
    for elevation in cfg.elevations_deg:
        pairs: list[GsdPair] = []
        for camera in (vis, swir):
            pair = pixel_gsd(
                camera.width_px / 2,
                camera.height_px / 2,
                math.radians(elevation),
                r,
                v,
                utc,
                utc,
                _OMEGA_EARTH_RAD_S,
                _WGS84_A_M,
                _WGS84_F,
                camera,
                cfg.plume_height_m,
            )
            if pair is None:
                raise ValueError(f"elevation {elevation} deg does not intersect the Earth")
            pairs.append(pair)
        spec = BinSpec(
            bin_id=f"el{round(elevation):02d}",
            lateral_m=pairs[0].lateral_m,
            along_m=pairs[0].along_m,
            elevation_deg=float(elevation),
        )
        bin_hw(spec)
        out.append(CameraBin(spec=spec, swir_gsd=pairs[1]))
    if len({item.spec.bin_id for item in out}) != len(out):
        raise ValueError("elevations must round to unique bin names")
    return tuple(out)


def resize_bilinear(planes: np.ndarray, tile_hw: tuple[int, int]) -> np.ndarray:
    """Resize channel-major planes with pixel centers aligned over one window.

    Args:
        planes: np.ndarray[float, (C, h, w)].
        tile_hw: Output ``(H, W)``.

    Returns:
        np.ndarray[float32, (C, H, W)]: Output pixel ``i`` samples source
        coordinate ``(i + 0.5) * h / H - 0.5``, clamped to the source edge.
    """
    array = np.asarray(planes, dtype=np.float64)
    if array.ndim != 3 or min(array.shape) < 1 or min(tile_hw) < 1:
        raise ValueError("expected non-empty (C, h, w) planes and a positive size")
    for axis, size in ((1, tile_hw[0]), (2, tile_hw[1])):
        source = array.shape[axis]
        coords = np.clip((np.arange(size) + 0.5) * source / size - 0.5, 0.0, source - 1.0)
        low = np.floor(coords).astype(np.int64)
        high = np.minimum(low + 1, source - 1)
        frac = coords - low
        shape = [1, 1, 1]
        shape[axis] = size
        frac = frac.reshape(shape)
        array = np.take(array, low, axis=axis) * (1 - frac) + np.take(array, high, axis=axis) * frac
    return np.asarray(array, dtype=np.float32)


def gaussian_blur(planes: np.ndarray, sigma_hw: tuple[float, float]) -> np.ndarray:
    """Blur channel-major planes with a separable Gaussian, edge-reflected.

    Args:
        planes: np.ndarray[float, (C, H, W)].
        sigma_hw: Sigma along H and W in pixels; values below 1e-3 skip that axis.

    Returns:
        np.ndarray[float64, (C, H, W)]: Blurred planes; each kernel sums to 1.
    """
    array = np.asarray(planes, dtype=np.float64)
    for axis, sigma in ((1, sigma_hw[0]), (2, sigma_hw[1])):
        if sigma < 1e-3:
            continue
        radius = max(1, math.ceil(3 * sigma))
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
        pad = [(0, 0), (0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(array, pad, mode="reflect" if array.shape[axis] > radius else "edge")
        size = array.shape[axis]
        out = np.zeros_like(array)
        for index, weight in enumerate(kernel):
            out += weight * np.take(padded, np.arange(index, index + size), axis=axis)
        array = out
    return array


def sharpen_b1(stack_dn: np.ndarray, band_ids: Sequence[str], b1_native_m: float) -> np.ndarray:
    """Return a copy of the stack with B2 detail finer than B1 added to B1.

    Args:
        stack_dn: np.ndarray[(C, H, W)] native 10 m L2A counts.
        band_ids: Band id per plane; must contain B1 and B2.
        b1_native_m: B1 native resolution in metres.

    Returns:
        np.ndarray[float64, (C, H, W)]: ``B1 + B2 - lowpass(B2)``; other planes copied.

    Notes:
        Zenodo B1 is smoothly upsampled from 60 m, so ``lowpass`` is an area
        average to the B1 grid followed by a center-aligned bilinear resize.
    """
    ids = list(band_ids)
    out = np.asarray(stack_dn, dtype=np.float64).copy()
    b1, b2 = ids.index("B1"), ids.index("B2")
    height, width = out.shape[1:]
    factor = b1_native_m / NATIVE_GSD_M
    coarse_hw = (max(1, round(height / factor)), max(1, round(width / factor)))
    low = resize_bilinear(resample_area(out[b2 : b2 + 1], coarse_hw), (height, width))[0]
    out[b1] = np.maximum(out[b1] + out[b2] - low, 0.0)
    return out


def add_sensor_noise(
    planes: np.ndarray,
    full_well_e: float,
    read_noise_e: float,
    saturation_reflectance: float,
    bit_depth: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add shot, read, and quantization noise to reflectance planes.

    Args:
        planes: np.ndarray[float, (C, H, W)] reflectance.
        full_well_e: Electrons at ``saturation_reflectance``.
        read_noise_e: Read noise sigma in electrons.
        saturation_reflectance: Reflectance that fills the full well.
        bit_depth: ADC bits; full well maps to ``2**bit_depth - 1`` counts.
        rng: Random generator.

    Returns:
        np.ndarray[float64, (C, H, W)]: Noisy reflectance clipped to ``[0, sat]``.

    Notes:
        Electrons ``e = rho / sat * FW``; noise sigma ``sqrt(e + read**2)``.
        Exposure is fixed by ``sat``; there is no dark current term.
    """
    gain = full_well_e / saturation_reflectance
    electrons = np.clip(np.asarray(planes, dtype=np.float64), 0, None) * gain
    noisy = electrons + rng.standard_normal(electrons.shape) * np.sqrt(electrons + read_noise_e**2)
    levels = 2**bit_depth - 1
    counts = np.clip(np.round(noisy / full_well_e * levels), 0, levels)
    return np.asarray(counts * full_well_e / levels / gain, dtype=np.float64)


def scene_mix(stem: str, seed: int) -> float:
    """Return the SWIR mix in ``[0, 1)`` for a scene stem, stable across processes."""
    rng = np.random.default_rng([zlib.crc32(stem.encode("utf-8")), seed])
    return float(rng.random())


def swir_weights_for(cfg: PairedConfig, stem: str) -> tuple[Mapping[str, float], float | None]:
    """Return the SWIR weights for one scene and its mix (None when fixed)."""
    if not cfg.swir_randomize:
        return cfg.swir_weights, None
    mix = scene_mix(stem, cfg.seed)
    low, high = cfg.swir_weights_low, cfg.swir_weights_high
    keys = set(low) | set(high)
    blended = {k: (1 - mix) * low.get(k, 0.0) + mix * high.get(k, 0.0) for k in keys}
    return _normalized(blended), mix


def _diffraction_sigma_px(wavelength_m: float, f_number: float, pitch_m: float) -> float:
    """Return the Airy-core Gaussian sigma in detector pixels."""
    return _AIRY_SIGMA * wavelength_m * f_number / pitch_m


def render_scene(
    stack_dn: np.ndarray,
    band_ids: Sequence[str],
    vis_table: WeightTable,
    cfg: PairedConfig,
    camera_bin: CameraBin,
    swir_weights: Mapping[str, float],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render one native Zenodo stack as BLUE, GREEN, RED, SWIR on the VIS grid.

    Args:
        stack_dn: np.ndarray[(C, 120, 120)] native L2A counts.
        band_ids: Band id per plane.
        vis_table: VIS prism weights.
        cfg: Paired settings.
        camera_bin: Elevation bin with both GSDs.
        swir_weights: Sentinel-2 weights for the SWIR plane.
        rng: Noise generator; required when ``cfg.noise`` is true.

    Returns:
        np.ndarray[float32, (4, H, W)]: Unit reflectance on the ``bin_hw`` grid.

    Raises:
        ValueError: If a weight names a missing band or noise lacks ``rng``.
    """
    ids = list(band_ids)
    stack = (
        sharpen_b1(stack_dn, ids, cfg.b1_native_m)
        if cfg.b1_sharpen
        else np.asarray(stack_dn, dtype=np.float64)
    )
    vis_native = mix_prism(stack, ids, vis_table).astype(np.float64)  # (3, 120, 120)
    reflectance = np.clip(stack / _DN_SCALE, 0.0, 1.0)
    swir_native = np.zeros((1, *stack.shape[1:]), dtype=np.float64)
    for band_id, weight in swir_weights.items():
        if band_id not in ids:
            raise ValueError(f"SWIR weight names missing band {band_id}")
        swir_native[0] += weight * reflectance[ids.index(band_id)]
    spec = camera_bin.spec
    out_hw = bin_hw(spec)
    vis_sigma = _diffraction_sigma_px(cfg.vis_wavelength_m, cfg.vis_f_number, cfg.vis_pitch_m)
    vis_sigma = math.hypot(vis_sigma, cfg.extra_blur_px)
    vis = gaussian_blur(
        vis_native,
        (vis_sigma * spec.along_m / NATIVE_GSD_M, vis_sigma * spec.lateral_m / NATIVE_GSD_M),
    )
    vis_out = resample_area(vis, out_hw).astype(np.float64)
    swir_gsd = camera_bin.swir_gsd
    swir_sigma = _diffraction_sigma_px(cfg.swir_wavelength_m, cfg.swir_f_number, cfg.swir_pitch_m)
    swir_sigma = math.hypot(swir_sigma, cfg.extra_blur_px)
    swir = gaussian_blur(
        swir_native,
        (
            swir_sigma * swir_gsd.along_m / NATIVE_GSD_M,
            swir_sigma * swir_gsd.lateral_m / NATIVE_GSD_M,
        ),
    )
    swir_hw = (
        max(1, round(EXTENT_M / swir_gsd.along_m)),
        max(1, round(EXTENT_M / swir_gsd.lateral_m)),
    )
    swir_out = resample_area(swir, swir_hw).astype(np.float64)
    if cfg.noise:
        if rng is None:
            raise ValueError("noise requires a random generator")
        vis_out = add_sensor_noise(
            vis_out,
            cfg.vis_full_well_e,
            cfg.vis_read_noise_e,
            cfg.saturation_reflectance,
            cfg.bit_depth,
            rng,
        )
        swir_out = add_sensor_noise(
            swir_out,
            cfg.swir_full_well_e,
            cfg.swir_read_noise_e,
            cfg.saturation_reflectance,
            cfg.bit_depth,
            rng,
        )
    swir_on_vis = resize_bilinear(swir_out, out_hw)
    image = np.concatenate([vis_out, swir_on_vis.astype(np.float64)], axis=0)
    return np.asarray(np.clip(image, 0.0, 1.0), dtype=np.float32)


def scene_refs(
    tile: TileRef, cfg: PairedConfig, bins: Sequence[CameraBin]
) -> tuple[RawTileRef, ...]:
    """Return one row per camera bin for one scene.

    Args:
        tile: Archive index entry.
        cfg: Paired settings; decides whether a ``swir_mix`` tag is added.
        bins: Camera bins in output order.

    Returns:
        tuple[RawTileRef, ...]: Rows ``<stem>-<bin_id>`` with the VIS GSD and
        the bin elevation. ``gsd_nominal`` stays false because the image is
        resampled to exactly the row GSD.
    """
    _, mix = swir_weights_for(cfg, tile.stem)
    metadata = observation_metadata(tile)
    if mix is not None:
        tags = (*metadata.conditions, ConditionTag("swir_mix", f"{mix:.2f}"))
        metadata = replace(metadata, conditions=tags)
    return tuple(
        RawTileRef(
            tile_id=f"{tile.stem}-{item.spec.bin_id}",
            group_id=tile.location_id,
            label=float(tile.positive),
            has_mask=tile.polygons is not None,
            gsd=actual_gsd(item.spec),
            height=bin_hw(item.spec)[0],
            width=bin_hw(item.spec)[1],
            frame_id=None,
            grid_rc=None,
            bin_id=item.spec.bin_id,
            theta_g_deg=item.spec.elevation_deg,
            metadata=metadata,
        )
        for item in bins
    )


class ZenodoPairedSource:
    """Read every labeled scene once and emit it at every camera elevation bin.

    Labels, split groups, and masks follow ``ZenodoSource``. Rows carry the
    VIS GSD, the bin elevation as ``theta_g_deg``, and a ``swir_mix``
    condition when the SWIR mix is randomized.
    """

    name = "zenodo"
    band_names: tuple[str, ...] = PAIRED_BANDS
    domain = "unit"
    source_ref = "10.5281/zenodo.4250706"

    def __init__(
        self,
        images_tar: str | Path,
        labels_tar: str | Path,
        vis_table: WeightTable,
        cfg: PairedConfig,
    ) -> None:
        self.images_tar = Path(images_tar)
        self.vis_table = vis_table
        self.cfg = cfg
        self.band_names = cfg.bands
        self._band_index = [PAIRED_BANDS.index(b) for b in cfg.bands]
        self.weight_table_id = f"{vis_table.id}+{cfg.id}"
        self.camera_bins = camera_bins(cfg)
        self.bins = tuple(item.spec for item in self.camera_bins)
        self._index = build_index(self.images_tar, Path(labels_tar))
        self._refs = tuple(
            ref for tile in self._index.tiles for ref in scene_refs(tile, cfg, self.camera_bins)
        )

    def index(self) -> tuple[RawTileRef, ...]:
        """Return rows in image-archive order, bins innermost."""
        return self._refs

    def iter_tiles(self) -> Iterator[RawTile]:
        """Render each scene at every bin and rasterize masks on the VIS grid."""
        cursor = 0
        for tile, stack, descriptions in iter_stacks(self.images_tar, self._index.tiles):
            ids = verify_band_order(coerce_descriptions(descriptions)).ids
            native = to_native_stack(stack)
            weights, _ = swir_weights_for(self.cfg, tile.stem)
            rng = np.random.default_rng([zlib.crc32(tile.stem.encode("utf-8")), self.cfg.seed, 1])
            for item in self.camera_bins:
                shape = bin_hw(item.spec)
                image = render_scene(native, ids, self.vis_table, self.cfg, item, weights, rng)[
                    self._band_index
                ]
                mask = (
                    None
                    if tile.polygons is None
                    else rasterize_percent_mask(
                        tile.polygons,
                        shape,
                        source_hw=(int(stack.shape[1]), int(stack.shape[2])),
                        fitted_hw=(NATIVE_SIDE, NATIVE_SIDE),
                    )
                )
                yield RawTile(self._refs[cursor], image, mask)
                cursor += 1
        if cursor != len(self._refs):
            raise ValueError("Zenodo stream ended before its index")
