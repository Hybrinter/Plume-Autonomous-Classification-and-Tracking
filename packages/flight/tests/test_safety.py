"""Unit tests for flight.payload.gimbal.safety -- confidence gate, area gate, deadband, rate.

REQ-AIML-DATA-008, REQ-AIML-DATA-009, REQ-AIML-GIMB-005, REQ-AIML-GIMB-006, REQ-AIML-GIMB-007
"""

# third-party
import pytest

# pact types
from flight.libs.messages import BlobMeta
from flight.libs.types import Err, FaultCode, Ok

# module under test
from flight.payload.gimbal import (
    apply_confidence_gate,
    apply_min_area_gate,
    check_deadband,
    check_rate_limit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_blob(
    blob_id: int = 1,
    mean_confidence: float = 0.85,
    pixel_area: int = 200,
    persistence_count: int = 1,
) -> BlobMeta:
    """Construct a BlobMeta for safety gate tests."""
    return BlobMeta(
        blob_id=blob_id,
        bbox=(0, 0, 20, 20),
        centroid_raw=(10.0, 10.0),
        pixel_area=pixel_area,
        mean_confidence=mean_confidence,
        persistence_count=persistence_count,
    )


# ---------------------------------------------------------------------------
# apply_confidence_gate tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confidence,expected_count", [
    (0.30, 0),    # well below gate (0.55)
    (0.54, 0),    # just below gate -- 0.55 - 0.01
    (0.55, 1),    # at gate exactly -- should pass
    (0.90, 1),    # well above gate
])
def test_confidence_gate_filters_low_confidence(
    confidence: float,
    expected_count: int,
) -> None:
    """Blobs with mean_confidence < 0.55 must be rejected by apply_confidence_gate."""
    blobs = (make_blob(mean_confidence=confidence),)
    result = apply_confidence_gate(blobs, threshold=0.55)
    assert len(result) == expected_count, (
        f"confidence={confidence}: expected {expected_count} blobs, got {len(result)}"
    )


def test_confidence_gate_passes_all_above_threshold() -> None:
    """All blobs above threshold must pass through."""
    blobs = (
        make_blob(blob_id=1, mean_confidence=0.60),
        make_blob(blob_id=2, mean_confidence=0.80),
        make_blob(blob_id=3, mean_confidence=1.00),
    )
    result = apply_confidence_gate(blobs, threshold=0.55)
    assert len(result) == 3


def test_confidence_gate_empty_input() -> None:
    """Empty input tuple must return empty output tuple."""
    result = apply_confidence_gate((), threshold=0.55)
    assert result == ()


# ---------------------------------------------------------------------------
# apply_min_area_gate tests
# ---------------------------------------------------------------------------


def test_min_area_gate_filters_small_blobs() -> None:
    """Blobs with pixel_area < min_px must be rejected."""
    small = make_blob(blob_id=1, pixel_area=10)   # below min of 15
    large = make_blob(blob_id=2, pixel_area=100)  # above min of 15
    result = apply_min_area_gate((small, large), min_px=15)
    assert len(result) == 1
    assert result[0].blob_id == 2, "Small blob should have been filtered out"


@pytest.mark.parametrize("area,min_px,expected_count", [
    (5, 15, 0),    # well below
    (14, 15, 0),   # one below threshold
    (15, 15, 1),   # at threshold exactly -- should pass (>= min_px)
    (100, 15, 1),  # well above
])
def test_min_area_gate_boundary(area: int, min_px: int, expected_count: int) -> None:
    """Parametrized boundary test for apply_min_area_gate."""
    blobs = (make_blob(pixel_area=area),)
    result = apply_min_area_gate(blobs, min_px=min_px)
    assert len(result) == expected_count, (
        f"area={area}, min_px={min_px}: expected {expected_count} blobs, got {len(result)}"
    )


# ---------------------------------------------------------------------------
# check_deadband tests
# ---------------------------------------------------------------------------


def test_deadband_below_min_returns_false() -> None:
    """Displacement below min_px returns Ok(False) -- no command needed, not a fault."""
    result = check_deadband(displacement_px=5.0, min_px=20, max_px=250)
    assert isinstance(result, Ok), f"Expected Ok, got {result}"
    assert result.value is False, (
        f"Expected Ok(False) for displacement below min, got Ok({result.value})"
    )


def test_deadband_within_range_returns_true() -> None:
    """Displacement within [min_px, max_px] returns Ok(True) -- command should be issued."""
    result = check_deadband(displacement_px=100.0, min_px=20, max_px=250)
    assert isinstance(result, Ok), f"Expected Ok, got {result}"
    assert result.value is True, (
        f"Expected Ok(True) for displacement in deadband range, got Ok({result.value})"
    )


def test_deadband_above_max_returns_error() -> None:
    """Displacement above max_px returns Err(GIMBAL_RUNAWAY) -- this is a fault condition."""
    result = check_deadband(displacement_px=300.0, min_px=20, max_px=250)
    assert isinstance(result, Err), (
        f"Expected Err, got Ok({result.value if hasattr(result, 'value') else result})"
    )
    assert result.error == FaultCode.GIMBAL_RUNAWAY, (
        f"Expected GIMBAL_RUNAWAY fault, got {result.error}"
    )


@pytest.mark.parametrize("displacement,expected_ok,expected_value_or_fault", [
    (0.0, True, False),         # below min (20)
    (19.9, True, False),        # just below min
    (20.0, True, True),         # at min exactly
    (125.0, True, True),        # midpoint
    (249.9, True, True),        # just below max
    (250.0, True, True),        # at max exactly -- still Ok(True) or Err? impl-defined
    (250.1, False, FaultCode.GIMBAL_RUNAWAY),  # just above max
    (500.0, False, FaultCode.GIMBAL_RUNAWAY),  # well above max
])
def test_deadband_parametrized(
    displacement: float,
    expected_ok: bool,
    expected_value_or_fault: object,
) -> None:
    """Full parametrized boundary sweep for check_deadband."""
    result = check_deadband(displacement_px=displacement, min_px=20, max_px=250)
    if expected_ok:
        assert isinstance(result, Ok), (
            f"displacement={displacement}: expected Ok, got Err({getattr(result, 'error', result)})"
        )
        assert result.value == expected_value_or_fault, (
            f"displacement={displacement}: expected Ok({expected_value_or_fault}), "
            f"got Ok({result.value})"
        )
    else:
        assert isinstance(result, Err), (
            f"displacement={displacement}: expected Err, got Ok({getattr(result, 'value', result)})"
        )
        assert result.error == expected_value_or_fault, (
            f"displacement={displacement}: expected Err({expected_value_or_fault}), "
            f"got Err({result.error})"
        )


# ---------------------------------------------------------------------------
# check_rate_limit tests
# ---------------------------------------------------------------------------


def test_rate_limit_blocks_too_fast() -> None:
    """A second command within the rate interval must be blocked (returns False)."""
    # rate_limit_hz=0.5 -> minimum interval = 2.0 seconds
    last_cmd = 100.0
    now = 100.5   # only 0.5s elapsed -- below the 2.0s minimum
    allowed = check_rate_limit(last_command_time=last_cmd, now=now, rate_limit_hz=0.5)
    assert allowed is False, (
        f"Expected False (rate limited) at 0.5s interval for 0.5Hz limit, got {allowed}"
    )


def test_rate_limit_allows_after_interval() -> None:
    """A command after the full rate interval has elapsed must be allowed (returns True)."""
    # rate_limit_hz=0.5 -> minimum interval = 2.0 seconds
    last_cmd = 100.0
    now = 102.5   # 2.5s elapsed -- above the 2.0s minimum
    allowed = check_rate_limit(last_command_time=last_cmd, now=now, rate_limit_hz=0.5)
    assert allowed is True, (
        f"Expected True (allowed) at 2.5s elapsed for 0.5Hz limit, got {allowed}"
    )


@pytest.mark.parametrize("elapsed,rate_hz,expected", [
    (0.5, 1.0, False),   # 0.5s < 1.0s minimum -- blocked
    (0.99, 1.0, False),  # just below 1.0s minimum -- blocked
    (1.0, 1.0, True),    # at exactly 1.0s -- allowed
    (2.0, 1.0, True),    # well above minimum -- allowed
    (0.1, 0.5, False),   # 0.5Hz -> 2.0s minimum; 0.1s elapsed -> blocked
    (2.0, 0.5, True),    # 2.0s elapsed at 0.5Hz minimum -- allowed
])
def test_rate_limit_boundary(elapsed: float, rate_hz: float, expected: bool) -> None:
    """Parametrized boundary test for check_rate_limit at various rates and elapsed times."""
    last_cmd = 1000.0
    now = last_cmd + elapsed
    result = check_rate_limit(last_command_time=last_cmd, now=now, rate_limit_hz=rate_hz)
    assert result == expected, (
        f"elapsed={elapsed}, rate_hz={rate_hz}: expected {expected}, got {result}"
    )
