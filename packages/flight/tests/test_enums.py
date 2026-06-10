"""Smoke tests for the migrated flight enums."""

from flight.libs.types import Band, FaultCode, GimbalState, SystemMode


def test_enum_value_mirrors_name() -> None:
    """Enum string values mirror their member names (log readability convention)."""
    assert SystemMode.IDLE.value == "IDLE"
    assert GimbalState.TRACKING.value == "TRACKING"


def test_faultcode_has_expected_members() -> None:
    """FaultCode exposes the known fault conditions used across subsystems."""
    names = {member.name for member in FaultCode}
    assert {"NONE", "MODEL_CORRUPT", "PROCESS_DIED", "WATCHDOG_EXPIRE"} <= names


def test_band_values_mirror_names() -> None:
    """Band enum string values must mirror member names."""
    for member in Band:
        assert member.value == member.name


def test_new_fault_codes_exist() -> None:
    """Ingest-chain fault codes are defined with name-mirroring values."""
    assert FaultCode.CALIBRATION_INVALID.value == "CALIBRATION_INVALID"
    assert FaultCode.FRAME_MALFORMED.value == "FRAME_MALFORMED"
