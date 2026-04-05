"""Unit tests for pact.preprocessing.quality — compute_quality_flags().

Satisfies: §6.2 of PACT_SW_ARCH.md — Preprocessing subsystem unit tests.
REQ-AIML-PREP-001, REQ-AIML-DATA-003
"""

from __future__ import annotations

# third-party
import numpy as np
import pytest

# module under test
from pact.preprocessing.quality import compute_quality_flags

# pact types
from pact.types.config import PreprocessingConfig
from pact.types.enums import FrameUsabilityTag

# Shared test constants — keep exposure_us below motion_smear threshold (default 5000µs)
# and set a high motion_smear threshold so saturation tests isolate only SATURATED flag.
_TS: str = "2026-04-03T00:00:00.000Z"
_CFG: PreprocessingConfig = PreprocessingConfig(motion_smear_exposure_us=20_000.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bands(shape: tuple[int, int, int] = (4, 256, 256), value: float = 0.0) -> np.ndarray:
    """Return a uniform float32 band array."""
    return np.full(shape, value, dtype=np.float32)  # np.ndarray[float32, (C, H, W)]


def _inject_saturation(
    bands: np.ndarray,
    fraction: float,
    sat_value: float = 1.0,
) -> np.ndarray:
    """Set `fraction` of pixels in band 0 to `sat_value` (>0.95 triggers SATURATED)."""
    bands = bands.copy()
    n_pixels = bands.shape[1] * bands.shape[2]
    n_sat = int(n_pixels * fraction)
    flat = bands[0].ravel()
    flat[:n_sat] = sat_value
    bands[0] = flat.reshape(bands.shape[1], bands.shape[2])
    return bands


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_flags_clean_frame() -> None:
    """A zeros array must return an empty frozenset (no flags raised)."""
    bands = _make_bands(value=0.0)
    flags = compute_quality_flags(bands, exposure_us=10_000.0, utc_timestamp=_TS, cfg=_CFG)
    assert flags == frozenset(), f"Expected no flags for clean frame, got {flags}"


def test_saturated_flag_raised() -> None:
    """An array with >5% pixels above 0.95 must raise the SATURATED flag."""
    bands = _make_bands(value=0.0)
    # Inject 10% saturated pixels — well above the 5% threshold
    bands = _inject_saturation(bands, fraction=0.10, sat_value=1.0)
    flags = compute_quality_flags(bands, exposure_us=10_000.0, utc_timestamp=_TS, cfg=_CFG)
    assert FrameUsabilityTag.SATURATED in flags, (
        f"Expected SATURATED flag for 10% saturated pixels, got {flags}"
    )


@pytest.mark.parametrize("fraction,expect_saturated", [
    (0.01, False),   # well below 5% threshold
    (0.049, False),  # just below 5% (4.9%) — should NOT trigger
    (0.05, False),   # at 5% threshold exactly — implementation uses strict >, so does NOT trigger
    (0.10, True),    # well above 5% — definitely triggers
])
def test_saturated_flag_boundary(fraction: float, expect_saturated: bool) -> None:
    """Parametrized boundary test for the SATURATED flag at the 5% pixel threshold.

    Note: the arch spec says '>5%'. Whether the boundary is strict or inclusive
    depends on the implementation. The test at 0.05 will reveal the actual boundary.
    Adjust expected value if implementation uses '>=' vs '>'.
    """
    bands = _make_bands(value=0.0)
    bands = _inject_saturation(bands, fraction=fraction, sat_value=1.0)
    flags = compute_quality_flags(bands, exposure_us=10_000.0, utc_timestamp=_TS, cfg=_CFG)
    if expect_saturated:
        assert FrameUsabilityTag.SATURATED in flags, (
            f"Expected SATURATED at fraction={fraction}, but flag was absent. Flags: {flags}"
        )
    else:
        assert FrameUsabilityTag.SATURATED not in flags, (
            f"Did not expect SATURATED at fraction={fraction}, but flag was present. Flags: {flags}"
        )


def test_quality_flags_returns_frozenset() -> None:
    """compute_quality_flags must always return a frozenset."""
    bands = _make_bands()
    flags = compute_quality_flags(bands, exposure_us=10_000.0, utc_timestamp=_TS, cfg=_CFG)
    assert isinstance(flags, frozenset), f"Expected frozenset, got {type(flags)}"


def test_flags_contain_only_usability_tags() -> None:
    """All elements in the returned frozenset must be FrameUsabilityTag members."""
    bands = _inject_saturation(_make_bands(), fraction=0.10)
    flags = compute_quality_flags(bands, exposure_us=10_000.0, utc_timestamp=_TS, cfg=_CFG)
    for flag in flags:
        assert isinstance(flag, FrameUsabilityTag), (
            f"Non-FrameUsabilityTag found in quality flags: {flag!r}"
        )
