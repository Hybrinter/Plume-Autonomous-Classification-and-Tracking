"""Command dictionary lookup + parameter-schema validation tests."""

from flight.libs.commands import COMMAND_DICTIONARY, lookup_command, validate_command
from flight.libs.types import CommandId, Err, FaultCode, Ok


def test_every_command_id_has_a_spec() -> None:
    """Every CommandId member has an entry in COMMAND_DICTIONARY."""
    for command_id in CommandId:
        assert command_id in COMMAND_DICTIONARY


def test_lookup_known_command() -> None:
    """lookup_command resolves a valid opcode string to its spec."""
    result = lookup_command("PING")
    assert isinstance(result, Ok)
    assert result.value.command_id is CommandId.PING


def test_lookup_unknown_command_rejected() -> None:
    """lookup_command returns Err for an unrecognised opcode."""
    result = lookup_command("NOT_A_COMMAND")
    assert isinstance(result, Err)
    assert result.error is FaultCode.COMMAND_INVALID


def test_validate_accepts_correct_params() -> None:
    """validate_command accepts params that exactly match the spec schema."""
    r = lookup_command("SET_THERMAL_LIMIT")
    assert isinstance(r, Ok)
    assert isinstance(validate_command(r.value, {"limit_c": 70.0}), Ok)


def test_validate_rejects_missing_param() -> None:
    """validate_command rejects an empty dict when a param is required."""
    r = lookup_command("SET_THERMAL_LIMIT")
    assert isinstance(r, Ok)
    result = validate_command(r.value, {})
    assert isinstance(result, Err)
    assert result.error is FaultCode.COMMAND_INVALID


def test_validate_rejects_wrong_type() -> None:
    """validate_command rejects a string where a float is required."""
    r = lookup_command("SET_THERMAL_LIMIT")
    assert isinstance(r, Ok)
    assert isinstance(validate_command(r.value, {"limit_c": "hot"}), Err)


def test_validate_rejects_unexpected_param() -> None:
    """validate_command rejects extra params not declared in the spec."""
    r = lookup_command("PING")
    assert isinstance(r, Ok)
    assert isinstance(validate_command(r.value, {"extra": 1}), Err)
