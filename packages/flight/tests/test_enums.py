"""Smoke tests for the migrated flight enums."""

from flight.libs.types import (
    AckStatus,
    Band,
    CommandId,
    FaultCode,
    GimbalCommandMode,
    GimbalState,
    LinkState,
    MessageType,
    ParamKind,
    SystemMode,
)


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


def test_gimbal_command_mode_values_mirror_names() -> None:
    """GimbalCommandMode string values must mirror member names."""
    for member in GimbalCommandMode:
        assert member.value == member.name
    assert {m.name for m in GimbalCommandMode} == {"RATE", "ABSOLUTE", "STOW", "HOME"}


def test_gimbal_fault_code_exists() -> None:
    """Driver-level gimbal failures have their own fault code."""
    assert FaultCode.GIMBAL_FAULT.value == "GIMBAL_FAULT"


def test_link_state_values_mirror_names() -> None:
    """LinkState string values mirror member names."""
    for member in LinkState:
        assert member.value == member.name


def test_ack_status_values_mirror_names() -> None:
    """AckStatus string values mirror member names."""
    for member in AckStatus:
        assert member.value == member.name


def test_command_id_values_mirror_names() -> None:
    """CommandId string values mirror member names."""
    for member in CommandId:
        assert member.value == member.name


def test_param_kind_values_mirror_names() -> None:
    """ParamKind string values mirror member names."""
    for member in ParamKind:
        assert member.value == member.name


def test_new_message_types_present() -> None:
    """The command-ack and link-state discriminants exist."""
    assert MessageType.COMMAND_ACK.value == "COMMAND_ACK"
    assert MessageType.LINK_STATE.value == "LINK_STATE"


def test_new_command_fault_codes_present() -> None:
    """The command-ingress fault codes exist."""
    for name in ("COMMAND_CRC_FAIL", "COMMAND_AUTH_FAIL", "COMMAND_SEQ_ERROR", "COMMAND_INVALID"):
        assert FaultCode[name].value == name
