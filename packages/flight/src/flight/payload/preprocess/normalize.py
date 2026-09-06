"""DN -> [0, 1] normalization for calibrated band planes.

normalized = clip(dn / full_scale, 0, 1), with full_scale the maximum ADC code the
sensor driver delivers. This is the reflectance-like domain the quality thresholds and
the model input contract assume (the model manifest's input domain is exactly this
function's output). Clipping bounds calibration under/overshoot; saturation detection on
the normalized planes still works because saturated pixels land at 1.0.

Full scale: by default 2**bit_depth - 1 (4095 for the 12-bit IMX264 ADC). The Spinnaker
driver can also deliver a 12-bit sample left-aligned in a 16-bit word (Mono16, max
65520), in which case the bit depth alone gives the wrong full scale; adc_max_dn lets the
caller pass the actual code maximum for the configured pixel format.

Domain note: DN / full_scale is a fraction of the ADC range at the frame's exposure and
gain, not a reflectance. It matches the Sentinel-2 reflectance domain the model was
trained on only when exposure and gain are held at the values the flight calibration
was built for; scale_to_reference_exposure removes the exposure/gain dependence when a
frame was taken at a different operating point.

Contains:
  - full_scale_dn: the ADC code maximum from bit depth or an explicit override.
  - normalize_dn: scale calibrated DN values by full scale and clip to [0, 1] float32.
  - scale_to_reference_exposure: rescale planes taken at (exposure, gain) to the
    signal they would have at a reference (exposure, gain).

Satisfies: REQ-AIML-PREP-002.
"""

from __future__ import annotations

# third-party
import numpy as np


def full_scale_dn(bit_depth: int, adc_max_dn: int | None = None) -> float:
    """Return the ADC code maximum used as the normalization full scale.

    Inputs:
        bit_depth (int): ADC bit depth; the default full scale is 2**bit_depth - 1
            (e.g. 4095 for 12-bit).
        adc_max_dn (int | None): Explicit code maximum for the configured pixel format
            (e.g. 65520 for a 12-bit sample left-aligned in Mono16). When given it
            overrides the bit-depth-derived value.

    Outputs:
        float: The full scale in DN.
    """
    if adc_max_dn is not None:
        return float(adc_max_dn)
    return float(2**bit_depth - 1)


def normalize_dn(
    planes: np.ndarray,
    bit_depth: int,
    adc_max_dn: int | None = None,
) -> np.ndarray:
    """Normalize calibrated DN band planes to [0, 1] float32 by ADC full scale.

    Divides every element by full_scale_dn(bit_depth, adc_max_dn) then clips to [0, 1].
    Values below zero arise from dark-subtraction overshoot and are clipped to 0.0;
    values above full scale arise from saturation/calibration artefacts and are clipped
    to 1.0. Saturation detection downstream still works because saturated pixels land
    at 1.0.

    Inputs:
        planes (np.ndarray[float32, (C, H, W)]): Calibrated DN values.
        bit_depth (int): ADC bit depth; full scale is 2**bit_depth - 1 unless
            adc_max_dn is given.
        adc_max_dn (int | None): Explicit ADC code maximum for the configured pixel
            format; overrides bit_depth when given.

    Outputs:
        np.ndarray[float32, (C, H, W)]: All values in [0, 1].

    Notes:
        The output dtype is always float32 regardless of the input dtype.
        This function is a pure transformation: no I/O, no global state.
    """
    full_scale = full_scale_dn(bit_depth, adc_max_dn)
    return np.clip(planes / full_scale, 0.0, 1.0).astype(np.float32)


def scale_to_reference_exposure(
    planes: np.ndarray,
    exposure_us: float,
    gain_db: float,
    reference_exposure_us: float,
    reference_gain_db: float,
) -> np.ndarray:
    """Rescale dark-corrected planes to the signal level of a reference operating point.

    After dark subtraction the remaining signal is photoelectrons times linear gain,
    and photoelectrons are proportional to exposure time, so

        signal_ref = signal * (t_ref / t) * 10**((g_ref - g) / 20)

    with analog gain in dB converted to a linear voltage ratio. Applying this before
    normalize_dn makes the normalized value independent of the frame's exposure and
    gain, so a frame taken at a shorter exposure lands in the same domain as the
    reference the calibration and model contract were built for.

    Inputs:
        planes (np.ndarray[float32, (C, H, W)]): Dark-corrected DN planes.
        exposure_us (float): Exposure the frame was taken at, microseconds (> 0).
        gain_db (float): Analog gain the frame was taken at, dB.
        reference_exposure_us (float): Reference exposure, microseconds (> 0).
        reference_gain_db (float): Reference analog gain, dB.

    Outputs:
        np.ndarray[float32, (C, H, W)]: Rescaled planes. Returned unchanged (as float32)
            when either exposure is non-positive, because the ratio is undefined.

    Notes:
        Valid only for dark-corrected (bias-free) signal; the bias offset does not scale
        with exposure. Clipping is not applied here; normalize_dn clips afterwards.
    """
    if exposure_us <= 0.0 or reference_exposure_us <= 0.0:
        return planes.astype(np.float32, copy=False)
    exposure_ratio = reference_exposure_us / exposure_us
    gain_ratio = 10.0 ** ((reference_gain_db - gain_db) / 20.0)
    return (planes * (exposure_ratio * gain_ratio)).astype(np.float32)
