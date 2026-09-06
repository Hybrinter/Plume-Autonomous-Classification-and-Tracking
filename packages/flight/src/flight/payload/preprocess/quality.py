"""flight.payload.preprocess.quality -- Per-frame quality flags and the keep/drop decision.

Satisfies: REQ-AIML-IMAG-002, REQ-AIML-DATA-003, REQ-AIML-DATA-005

Computes the FrameUsabilityTag flags for one calibrated, normalized frame, the numeric
metrics behind them, and the keep/drop verdict (TRAINING / TRACKING / INVALID) that
decides whether the frame goes to the model interface and whether it is fit for the
training dataset. Every function is pure.

Flag conditions:
    INCOMPLETE_METADATA -- nonpositive exposure, missing timestamp, or (when given) a
                           negative gain.
    SATURATED           -- more than cfg.saturation_fraction_threshold of the pixels of
                           any band sit above the saturation level. Measured on the RAW
                           mosaic per CFA cell class when raw_mosaic and adc_max_dn are
                           supplied (saturation is an ADC property: flat-field division
                           and clipping can hide a saturated pixel or fake one), else on
                           the normalized planes against SATURATION_PIXEL_LEVEL.
    MOTION_SMEAR        -- predicted smear in band-plane pixels
                           smear_px = (|gimbal slew| + platform rate) * t_exp / IFOV
                           exceeds cfg.max_motion_smear_px. The platform rate is the
                           ISS ground-track angular rate (~1 deg/s at nadir from 420 km),
                           which smears the scene even with the gimbal still; it is 0.0
                           unless the caller supplies it.
    CLOUD_CONTAMINATED  -- default (legacy): mean NIR / mean RED exceeds
                           cfg.nir_red_ratio_threshold. Physics caveat: with the flight
                           passbands (665 / 842 nm) a high NIR/RED ratio is the
                           vegetation signature, while cloud is bright and spectrally
                           flat (ratio ~1). Pass a CloudTest to use the bright-and-flat
                           fraction test instead.
    SUNGLINT            -- mean NIR exceeds cfg.sunglint_nir_mean_threshold. Over land
                           this is an over-brightness (exposure) gate rather than glint.

Band identity is resolved BY NAME through band_names (default BLUE, GREEN, RED, NIR),
never by fixed index, so a reordered InferenceConfig.input_bands cannot silently swap
the RED and NIR planes the spectral checks read. When RED or NIR is absent from
band_names the spectral checks cannot run: their metrics are NaN and their flags are
not raised.

Contains:
  - SATURATION_PIXEL_LEVEL: default saturation level as a fraction of full scale.
  - DEFAULT_BAND_NAMES: the 4-band order assumed when band_names is not given.
  - CloudTest: thresholds for the bright-and-flat cloud fraction test.
  - QualityMetrics: the numbers behind the flags, for telemetry and tests.
  - UsabilityPolicy / DEFAULT_USABILITY_POLICY: which flags make a frame INVALID and
    which make it TRACKING-only.
  - saturated_fraction_normalized / saturated_fraction_raw: per-band saturated fraction.
  - predicted_smear_px: gimbal plus platform smear length in band-plane pixels.
  - cloud_fraction: fraction of bright, spectrally flat pixels.
  - compute_quality_metrics: evaluate every metric; Err(FRAME_MALFORMED) on bad shapes.
  - flags_from_metrics: raise flags from metrics and thresholds.
  - compute_quality_flags: metrics -> flags in one call (legacy entry point).
  - decide_usability: flags -> TRAINING | TRACKING | INVALID under a policy.
"""

from __future__ import annotations

# stdlib
import math
from dataclasses import dataclass
from typing import Final

# third-party
import numpy as np

# internal
from flight.libs.config import PreprocessingConfig
from flight.libs.types import Err, FaultCode, FrameUsabilityTag, Ok, Result
from flight.payload.preprocess.band_select import band_index
from flight.payload.preprocess.demosaic import separate_bands

# Saturation level as a fraction of full scale; the same level is used on normalized
# planes and (times adc_max_dn) on the raw mosaic.
SATURATION_PIXEL_LEVEL: Final[float] = 0.95

DEFAULT_BAND_NAMES: Final[tuple[str, ...]] = ("BLUE", "GREEN", "RED", "NIR")

_BAND_RED: Final[str] = "RED"
_BAND_NIR: Final[str] = "NIR"
_RATIO_EPSILON: Final[float] = 1e-6


@dataclass(frozen=True, slots=True)
class CloudTest:
    """Thresholds for the bright-and-spectrally-flat cloud fraction test.

    A pixel is cloud-like when every band exceeds bright_threshold AND its spectral
    spread (max - min) / mean is below whiteness_tolerance. The frame is flagged when
    the fraction of cloud-like pixels exceeds fraction_threshold.

    Attributes:
        bright_threshold: float minimum normalized value in every band.
        whiteness_tolerance: float maximum (max - min) / mean across bands.
        fraction_threshold: float fraction of cloud-like pixels that raises the flag.
    """

    bright_threshold: float
    whiteness_tolerance: float
    fraction_threshold: float


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Numeric quality metrics for one frame.

    Attributes:
        saturated_fraction: tuple[float, ...] saturated pixel fraction per band. In
            band_names order when measured on normalized planes; in CELL_OFFSETS order
            (one entry per CFA cell class) when measured on the raw mosaic.
        saturation_on_raw: bool True when saturated_fraction came from the raw mosaic.
        smear_px: float predicted smear length in band-plane pixels.
        nir_red_ratio: float mean NIR / mean RED; NaN when either band is absent.
        cloud_fraction: float fraction of bright-and-flat pixels; NaN when no CloudTest
            was supplied.
        nir_mean: float mean normalized NIR; NaN when NIR is absent.
        metadata_complete: bool False when exposure, timestamp, or gain is invalid.
    """

    saturated_fraction: tuple[float, ...]
    saturation_on_raw: bool
    smear_px: float
    nir_red_ratio: float
    cloud_fraction: float
    nir_mean: float
    metadata_complete: bool


@dataclass(frozen=True, slots=True)
class UsabilityPolicy:
    """Which quality flags drop a frame and which restrict it to tracking use.

    Attributes:
        invalid_flags: frozenset[FrameUsabilityTag] any of these -> INVALID; the frame
            is skipped by the model interface.
        tracking_only_flags: frozenset[FrameUsabilityTag] any of these (and none of
            invalid_flags) -> TRACKING; the model runs for control but the frame is
            excluded from the training dataset.
    """

    invalid_flags: frozenset[FrameUsabilityTag]
    tracking_only_flags: frozenset[FrameUsabilityTag]


# Recommended default: metadata, saturation, cloud and over-brightness corrupt both the
# detection and the training label; a few pixels of smear still give a valid centroid
# for control but blur segmentation boundaries, so smeared frames track but do not train.
DEFAULT_USABILITY_POLICY: Final[UsabilityPolicy] = UsabilityPolicy(
    invalid_flags=frozenset(
        {
            FrameUsabilityTag.INCOMPLETE_METADATA,
            FrameUsabilityTag.SATURATED,
            FrameUsabilityTag.CLOUD_CONTAMINATED,
            FrameUsabilityTag.SUNGLINT,
        }
    ),
    tracking_only_flags=frozenset({FrameUsabilityTag.MOTION_SMEAR}),
)


def saturated_fraction_normalized(
    bands: np.ndarray,
    level: float = SATURATION_PIXEL_LEVEL,
) -> tuple[float, ...]:
    """Fraction of pixels above level in each normalized band plane.

    Inputs:
        bands (np.ndarray[float32, (C, H, W)]): Normalized planes in [0, 1].
        level (float): Saturation level in normalized units (strict >).

    Outputs:
        tuple[float, ...]: One fraction per channel, in channel order.
    """
    n_pixels = bands.shape[1] * bands.shape[2]
    return tuple(float((bands[c] > level).sum()) / n_pixels for c in range(bands.shape[0]))


def saturated_fraction_raw(
    mosaic: np.ndarray,
    adc_max_dn: int,
    level_frac: float = SATURATION_PIXEL_LEVEL,
    cfa_phase: tuple[int, int] = (0, 0),
) -> Result[tuple[float, ...], FaultCode]:
    """Fraction of raw pixels above level_frac * adc_max_dn in each CFA cell class.

    Saturation is a property of the ADC code, so it is measured on the raw mosaic
    before dark/flat correction and before clipping, which can both hide a saturated
    pixel (flat > 1) or manufacture one (flat < 1). Each 2x2 cell class is one band, so
    the fractions are per band in CELL_OFFSETS order.

    Inputs:
        mosaic (np.ndarray[*, (H, W)]): Raw mosaic plane in DN.
        adc_max_dn (int): ADC code maximum (e.g. 4095 for 12-bit).
        level_frac (float): Saturation level as a fraction of adc_max_dn (strict >).
        cfa_phase (tuple[int, int]): First complete tile position, see demosaic.

    Outputs:
        Result[tuple[float, ...], FaultCode]:
            Ok(tuple of 4 floats) in CELL_OFFSETS order;
            Err(FaultCode.FRAME_MALFORMED) if the mosaic cannot be separated.
    """
    planes = separate_bands(mosaic, cfa_phase)
    if isinstance(planes, Err):
        return Err(planes.error)
    return Ok(saturated_fraction_normalized(planes.value, level_frac * adc_max_dn))


def predicted_smear_px(
    slew_rate_deg_per_s: float,
    exposure_us: float,
    ifov_deg_per_px: float,
    platform_rate_deg_per_s: float = 0.0,
) -> float:
    """Predicted motion smear length in band-plane pixels over one exposure.

        smear_px = (|slew| + platform_rate) * t_exp / IFOV

    The scene moves across the focal plane at the sum of the gimbal slew rate and the
    platform's angular rate relative to the ground (ISS ground track, ~1 deg/s at nadir
    from 420 km); a stationary gimbal does not mean a stationary scene. The magnitude of
    the slew is used so that a negative encoder rate smears exactly like a positive one.

    Inputs:
        slew_rate_deg_per_s (float): Gimbal slew rate over the exposure, deg/s (either
            sign). 0.0 when unknown.
        exposure_us (float): Exposure time, microseconds.
        ifov_deg_per_px (float): Band-plane IFOV, deg/px.
        platform_rate_deg_per_s (float): Platform ground-track angular rate, deg/s.
            Default 0.0 (gimbal-only smear).

    Outputs:
        float: Smear length in band-plane pixels; 0.0 when ifov is non-positive.
    """
    if ifov_deg_per_px <= 0.0:
        return 0.0
    total_rate = abs(slew_rate_deg_per_s) + platform_rate_deg_per_s
    return total_rate * (exposure_us * 1e-6) / ifov_deg_per_px


def cloud_fraction(bands: np.ndarray, test: CloudTest) -> float:
    """Fraction of pixels that are bright in every band and spectrally flat.

    Cloud in the VNIR (490-842 nm) is bright and nearly white; vegetation is dark in
    RED and bright in NIR (high spread), bare ground and water are dark. A pixel is
    cloud-like when min over bands > bright_threshold and (max - min) / mean <
    whiteness_tolerance.

    Inputs:
        bands (np.ndarray[float32, (C, H, W)]): Normalized planes in [0, 1], C >= 1.
        test (CloudTest): Thresholds.

    Outputs:
        float: Fraction of cloud-like pixels in [0, 1].
    """
    band_min = bands.min(axis=0)  # np.ndarray[float32, (H, W)]
    band_max = bands.max(axis=0)  # np.ndarray[float32, (H, W)]
    band_mean = bands.mean(axis=0)  # np.ndarray[float32, (H, W)]
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = np.where(band_mean > 0.0, (band_max - band_min) / band_mean, np.inf)
    cloud_like = (band_min > test.bright_threshold) & (spread < test.whiteness_tolerance)
    return float(cloud_like.mean())


def compute_quality_metrics(
    bands: np.ndarray,
    exposure_us: float,
    slew_rate_deg_per_s: float,
    ifov_deg_per_px: float,
    utc_timestamp: str,
    band_names: tuple[str, ...] = DEFAULT_BAND_NAMES,
    gain_db: float | None = None,
    raw_mosaic: np.ndarray | None = None,
    adc_max_dn: int | None = None,
    cfa_phase: tuple[int, int] = (0, 0),
    platform_rate_deg_per_s: float = 0.0,
    cloud_test: CloudTest | None = None,
    saturation_level: float = SATURATION_PIXEL_LEVEL,
) -> Result[QualityMetrics, FaultCode]:
    """Evaluate every per-frame quality metric.

    Inputs:
        bands (np.ndarray[float32, (C, H, W)]): Calibrated, normalized planes in
            band_names order.
        exposure_us (float): Exposure time, microseconds.
        slew_rate_deg_per_s (float): Gimbal slew rate over the exposure, deg/s.
        ifov_deg_per_px (float): Band-plane IFOV, deg/px.
        utc_timestamp (str): ISO 8601 frame timestamp; empty means missing.
        band_names (tuple[str, ...]): Channel names of bands. Default
            (BLUE, GREEN, RED, NIR).
        gain_db (float | None): Analog gain, dB; a negative value marks the metadata
            incomplete. None skips the check.
        raw_mosaic (np.ndarray | None): Raw (H_m, W_m) mosaic in DN. With adc_max_dn,
            saturation is measured here per CFA cell class instead of on bands.
        adc_max_dn (int | None): ADC code maximum for raw saturation.
        cfa_phase (tuple[int, int]): First complete tile position for raw saturation.
        platform_rate_deg_per_s (float): Platform ground-track angular rate, deg/s.
        cloud_test (CloudTest | None): Use the bright-and-flat cloud fraction test;
            None leaves cloud_fraction NaN (the legacy NIR/RED ratio is always computed).
        saturation_level (float): Saturation level as a fraction of full scale.

    Outputs:
        Result[QualityMetrics, FaultCode]:
            Ok(QualityMetrics);
            Err(FaultCode.FRAME_MALFORMED) if bands is not rank 3, its channel count
                differs from len(band_names), or raw_mosaic cannot be separated.
    """
    if bands.ndim != 3 or bands.shape[0] != len(band_names):
        return Err(FaultCode.FRAME_MALFORMED)

    metadata_complete = exposure_us > 0.0 and bool(utc_timestamp)
    if gain_db is not None and gain_db < 0.0:
        metadata_complete = False

    if raw_mosaic is not None and adc_max_dn is not None:
        raw_fraction = saturated_fraction_raw(raw_mosaic, adc_max_dn, saturation_level, cfa_phase)
        if isinstance(raw_fraction, Err):
            return Err(raw_fraction.error)
        saturated = raw_fraction.value
        on_raw = True
    else:
        saturated = saturated_fraction_normalized(bands, saturation_level)
        on_raw = False

    smear = predicted_smear_px(
        slew_rate_deg_per_s, exposure_us, ifov_deg_per_px, platform_rate_deg_per_s
    )

    red_idx = band_index(band_names, _BAND_RED)
    nir_idx = band_index(band_names, _BAND_NIR)
    nir_mean = math.nan
    ratio = math.nan
    if isinstance(nir_idx, Ok):
        nir_mean = float(bands[nir_idx.value].mean())
        if isinstance(red_idx, Ok):
            red_mean = float(bands[red_idx.value].mean())
            ratio = nir_mean / (red_mean + _RATIO_EPSILON)

    cloud = cloud_fraction(bands, cloud_test) if cloud_test is not None else math.nan

    return Ok(
        QualityMetrics(
            saturated_fraction=saturated,
            saturation_on_raw=on_raw,
            smear_px=smear,
            nir_red_ratio=ratio,
            cloud_fraction=cloud,
            nir_mean=nir_mean,
            metadata_complete=metadata_complete,
        )
    )


def flags_from_metrics(
    metrics: QualityMetrics,
    cfg: PreprocessingConfig,
    cloud_test: CloudTest | None = None,
) -> frozenset[FrameUsabilityTag]:
    """Raise quality flags from metrics and thresholds.

    Inputs:
        metrics (QualityMetrics): Output of compute_quality_metrics.
        cfg (PreprocessingConfig): Thresholds for saturation fraction, smear, NIR/RED
            ratio and NIR mean.
        cloud_test (CloudTest | None): When given, CLOUD_CONTAMINATED is decided by
            metrics.cloud_fraction > cloud_test.fraction_threshold; otherwise by
            metrics.nir_red_ratio > cfg.nir_red_ratio_threshold.

    Outputs:
        frozenset[FrameUsabilityTag]: The flags raised; empty means clean.

    Notes:
        A NaN metric (band absent, or no CloudTest) never raises its flag.
    """
    flags: set[FrameUsabilityTag] = set()
    if not metrics.metadata_complete:
        flags.add(FrameUsabilityTag.INCOMPLETE_METADATA)
    if any(f > cfg.saturation_fraction_threshold for f in metrics.saturated_fraction):
        flags.add(FrameUsabilityTag.SATURATED)
    if metrics.smear_px > cfg.max_motion_smear_px:
        flags.add(FrameUsabilityTag.MOTION_SMEAR)
    if cloud_test is not None:
        if metrics.cloud_fraction > cloud_test.fraction_threshold:
            flags.add(FrameUsabilityTag.CLOUD_CONTAMINATED)
    elif metrics.nir_red_ratio > cfg.nir_red_ratio_threshold:
        flags.add(FrameUsabilityTag.CLOUD_CONTAMINATED)
    if metrics.nir_mean > cfg.sunglint_nir_mean_threshold:
        flags.add(FrameUsabilityTag.SUNGLINT)
    return frozenset(flags)


def compute_quality_flags(
    bands: np.ndarray,
    exposure_us: float,
    slew_rate_deg_per_s: float,
    ifov_deg_per_px: float,
    utc_timestamp: str,
    cfg: PreprocessingConfig,
    band_names: tuple[str, ...] = DEFAULT_BAND_NAMES,
    gain_db: float | None = None,
    raw_mosaic: np.ndarray | None = None,
    adc_max_dn: int | None = None,
    cfa_phase: tuple[int, int] = (0, 0),
    platform_rate_deg_per_s: float = 0.0,
    cloud_test: CloudTest | None = None,
    saturation_level: float = SATURATION_PIXEL_LEVEL,
) -> frozenset[FrameUsabilityTag]:
    """Compute per-frame quality flags for a calibrated, normalized multispectral frame.

    Convenience wrapper: compute_quality_metrics followed by flags_from_metrics. An
    empty frozenset means the frame is clean and inference-ready.

    Inputs:
        bands (np.ndarray[float32, (C, H, W)]): Calibrated and normalized band array in
            band_names order.
        exposure_us (float): Camera exposure time in microseconds.
        slew_rate_deg_per_s (float): Gimbal slew rate over the exposure, deg/s (0.0
            when unknown -- the gimbal term of the smear gate degrades to zero).
        ifov_deg_per_px (float): Band-plane IFOV, deg/px (SensorConfig.ifov_deg_per_px).
        utc_timestamp (str): ISO 8601 timestamp string from the frame metadata.
        cfg (PreprocessingConfig): Quality-flag thresholds.
        band_names (tuple[str, ...]): Channel names of bands; default
            (BLUE, GREEN, RED, NIR).
        gain_db (float | None): Analog gain, dB; negative marks metadata incomplete.
        raw_mosaic (np.ndarray | None): Raw mosaic for raw-DN saturation (with
            adc_max_dn).
        adc_max_dn (int | None): ADC code maximum for raw-DN saturation.
        cfa_phase (tuple[int, int]): First complete tile position for raw saturation.
        platform_rate_deg_per_s (float): Platform ground-track angular rate, deg/s.
        cloud_test (CloudTest | None): Bright-and-flat cloud test; None uses the
            legacy NIR/RED ratio against cfg.nir_red_ratio_threshold.
        saturation_level (float): Saturation level as a fraction of full scale.

    Outputs:
        frozenset[FrameUsabilityTag]: The flags raised for this frame; empty if clean.
            frozenset({FrameUsabilityTag.INVALID}) when the input is malformed (wrong
            rank or channel count, or an unseparable raw mosaic): such a frame cannot
            be assessed and decide_usability maps it to INVALID.
    """
    metrics = compute_quality_metrics(
        bands,
        exposure_us,
        slew_rate_deg_per_s,
        ifov_deg_per_px,
        utc_timestamp,
        band_names,
        gain_db,
        raw_mosaic,
        adc_max_dn,
        cfa_phase,
        platform_rate_deg_per_s,
        cloud_test,
        saturation_level,
    )
    if isinstance(metrics, Err):
        return frozenset({FrameUsabilityTag.INVALID})
    return flags_from_metrics(metrics.value, cfg, cloud_test)


def decide_usability(
    flags: frozenset[FrameUsabilityTag],
    policy: UsabilityPolicy = DEFAULT_USABILITY_POLICY,
) -> FrameUsabilityTag:
    """Keep-or-drop decision: map raised flags to TRAINING, TRACKING, or INVALID.

    Precedence: INVALID (frame is skipped by the model interface) if the flags contain
    FrameUsabilityTag.INVALID or any of policy.invalid_flags; else TRACKING (model runs
    for control, frame excluded from training) if any of policy.tracking_only_flags;
    else TRAINING (clean frame, usable for both).

    Inputs:
        flags (frozenset[FrameUsabilityTag]): Output of compute_quality_flags.
        policy (UsabilityPolicy): Flag classification; default DEFAULT_USABILITY_POLICY.

    Outputs:
        FrameUsabilityTag: One of TRAINING, TRACKING, INVALID.
    """
    if FrameUsabilityTag.INVALID in flags or flags & policy.invalid_flags:
        return FrameUsabilityTag.INVALID
    if flags & policy.tracking_only_flags:
        return FrameUsabilityTag.TRACKING
    return FrameUsabilityTag.TRAINING
