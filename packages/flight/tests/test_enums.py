"""Smoke tests for the migrated flight enums."""

from flight.libs.types import FaultCode, GimbalState, SystemMode


def test_enum_value_mirrors_name() -> None:
    """Enum string values mirror their member names (log readability convention)."""
    assert SystemMode.IDLE.value == "IDLE"
    assert GimbalState.TRACKING.value == "TRACKING"


def test_faultcode_has_expected_members() -> None:
    """FaultCode exposes the known fault conditions used across subsystems."""
    names = {member.name for member in FaultCode}
    assert {"NONE", "MODEL_CORRUPT", "PROCESS_DIED", "WATCHDOG_EXPIRE"} <= names
