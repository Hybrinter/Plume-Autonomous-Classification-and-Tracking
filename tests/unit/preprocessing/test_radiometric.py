"""Unit tests for pact.preprocessing.radiometric — apply_calibration().

Satisfies: §6.2 of PACT_SW_ARCH.md — Preprocessing subsystem unit tests.
REQ-AIML-PREP-001, REQ-AIML-DATA-003
"""

from __future__ import annotations

# third-party
import numpy as np
import pytest

# module under test
from pact.preprocessing.radiometric import RadiometricCalibration, apply_calibration

# pact types
from pact.types.enums import Err, FaultCode, Ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_calibration(
    shape: tuple[int, int, int] = (4, 8, 8),
    dark_value: float = 0.05,
    flat_value: float = 1.0,
) -> RadiometricCalibration:
    """Return a RadiometricCalibration with uniform dark frame and flat field."""
    return RadiometricCalibration(
        dark_frame=np.full(shape, dark_value, dtype=np.float32),
        flat_field=np.full(shape, flat_value, dtype=np.float32),
    )


def _make_raw(shape: tuple[int, int, int] = (4, 8, 8), value: float = 0.5) -> np.ndarray:
    """Return a uniform float32 raw array."""
    return np.full(shape, value, dtype=np.float32)  # np.ndarray[float32, (C, H, W)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_calibration_returns_ok() -> None:
    """apply_calibration with valid inputs must return Ok wrapping a numpy array."""
    raw = _make_raw()
    cal = _make_calibration()
    result = apply_calibration(raw, cal)
    assert isinstance(result, Ok), f"Expected Ok, got Err({result.error if hasattr(result, 'error') else result})"


def test_calibration_nan_returns_err() -> None:
    """apply_calibration with NaN in dark frame must return Err(FaultCode.INFERENCE_NAN)."""
    raw = _make_raw()
    dark = np.full((4, 8, 8), 0.05, dtype=np.float32)
    dark[0, 0, 0] = float("nan")
    flat = np.ones((4, 8, 8), dtype=np.float32)
    cal = RadiometricCalibration(dark_frame=dark, flat_field=flat)
    result = apply_calibration(raw, cal)
    assert isinstance(result, Err), f"Expected Err, got Ok"
    assert result.error == FaultCode.INFERENCE_NAN, (
        f"Expected INFERENCE_NAN, got {result.error}"
    )


def test_dark_frame_subtraction_output_shape() -> None:
    """apply_calibration output shape must match the input raw array shape."""
    raw = _make_raw(shape=(4, 16, 16))
    cal = _make_calibration(shape=(4, 16, 16))
    result = apply_calibration(raw, cal)
    assert isinstance(result, Ok), f"Expected Ok, got {result}"
    assert result.value.shape == (4, 16, 16), (
        f"Output shape mismatch: expected (4,16,16), got {result.value.shape}"
    )


def test_calibration_dark_subtraction_values() -> None:
    """Dark frame subtraction must reduce pixel values by the dark frame amount."""
    raw = _make_raw(value=0.5)
    dark_value = 0.1
    cal = _make_calibration(dark_value=dark_value, flat_value=1.0)
    result = apply_calibration(raw, cal)
    assert isinstance(result, Ok), f"Expected Ok, got {result}"
    # After dark subtraction and flat-field divide: (0.5 - 0.1) / 1.0 = 0.4
    np.testing.assert_allclose(result.value, 0.4, atol=1e-5)


def test_calibration_inf_in_input_returns_err() -> None:
    """apply_calibration with inf in the raw input should return Err(INFERENCE_NAN)."""
    raw = np.full((4, 8, 8), float("inf"), dtype=np.float32)
    cal = _make_calibration()
    result = apply_calibration(raw, cal)
    # inf in input propagates to output → should be caught as NaN/Inf fault
    assert isinstance(result, Err), (
        "Expected Err for inf input, but got Ok — calibration must detect inf/nan outputs"
    )
