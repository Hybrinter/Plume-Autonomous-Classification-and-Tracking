"""Typed command dictionary (see flight.libs.commands.dictionary)."""

from flight.libs.commands.dictionary import (
    COMMAND_DICTIONARY,
    CommandSpec,
    ParamSpec,
    lookup_command,
    validate_command,
)

__all__ = [
    "COMMAND_DICTIONARY",
    "CommandSpec",
    "ParamSpec",
    "lookup_command",
    "validate_command",
]
