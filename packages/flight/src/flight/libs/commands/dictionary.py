"""Typed command dictionary: the validation authority for inbound ground commands (pure).

Maps each CommandId to a CommandSpec carrying its canonical target subsystem, its required
parameter schema (name + primitive kind), and whether it is hazardous (ARM/EXECUTE gated --
enforced by the command router in Phase 6B). Validation is data-driven iteration over the
spec's declared params: no callable dispatch tables and no getattr (honors the strong-typing
rule). The dictionary lives in flight.libs so both iss_iface and the composition root can
import it without a layering violation.

Contains:
  - ParamSpec / CommandSpec: frozen data describing one parameter / one command.
  - COMMAND_DICTIONARY: the CommandId -> CommandSpec registry.
  - lookup_command: resolve a wire command_id string to its spec (Result).
  - validate_command: check a params dict against a spec's schema (Result).

Satisfies: REQ-COMM-HIGH-003.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# internal
from flight.libs.types import CommandId, Err, FaultCode, Ok, ParamKind, Result

_KIND_TO_TYPE: dict[ParamKind, type] = {
    ParamKind.STR: str,
    ParamKind.INT: int,
    ParamKind.FLOAT: float,
    ParamKind.BOOL: bool,
}


@dataclass(slots=True, frozen=True)
class ParamSpec:
    """One required command parameter: its key and primitive kind.

    Note: ParamKind.FLOAT accepts int or float (ints widen to float); ParamKind.INT and
    ParamKind.BOOL are exact (bool is rejected where INT/FLOAT is required and vice versa,
    because bool is a subclass of int -- validate_command guards this explicitly).
    """

    name: str
    kind: ParamKind


@dataclass(slots=True, frozen=True)
class CommandSpec:
    """The dictionary entry for one command: target, schema, and hazard class.

    Args/fields:
        command_id: The opcode this spec describes.
        target: Canonical destination subsystem name (lowercase, e.g. "thermal"); iss_iface
            stamps CommandMsg.target from this, so the ground frame need not carry a target.
        params: The required parameters, in declaration order.
        hazardous: True if the command requires the ARM/EXECUTE two-step (Phase 6B). 6A
            commands are all non-hazardous.
    """

    command_id: CommandId
    target: str
    params: tuple[ParamSpec, ...]
    hazardous: bool


COMMAND_DICTIONARY: dict[CommandId, CommandSpec] = {
    CommandId.PING: CommandSpec(CommandId.PING, "core", (), hazardous=False),
    CommandId.NOOP: CommandSpec(CommandId.NOOP, "core", (), hazardous=False),
    CommandId.SET_THERMAL_LIMIT: CommandSpec(
        CommandId.SET_THERMAL_LIMIT,
        "thermal",
        (ParamSpec("limit_c", ParamKind.FLOAT),),
        hazardous=False,
    ),
    CommandId.EXIT_SAFE: CommandSpec(
        CommandId.EXIT_SAFE,
        "fault",
        (ParamSpec("phase", ParamKind.STR),),
        hazardous=True,
    ),
    CommandId.RELEASE_LAUNCH_LOCK: CommandSpec(
        CommandId.RELEASE_LAUNCH_LOCK,
        "mechanical",
        (ParamSpec("phase", ParamKind.STR),),
        hazardous=True,
    ),
    CommandId.UPLOAD_MODEL_CHUNK: CommandSpec(
        CommandId.UPLOAD_MODEL_CHUNK,
        "iss_iface",
        (
            ParamSpec("chunk_index", ParamKind.INT),
            ParamSpec("total_chunks", ParamKind.INT),
            ParamSpec("data_b64", ParamKind.STR),
            ParamSpec("crc32", ParamKind.INT),
        ),
        hazardous=False,
    ),
    CommandId.ACTIVATE_MODEL: CommandSpec(
        CommandId.ACTIVATE_MODEL,
        "model_deploy",
        (ParamSpec("version", ParamKind.STR),),
        hazardous=False,
    ),
}


def routable_targets() -> frozenset[str]:
    """Return the set of subsystem targets any command in the dictionary may be routed to.

    Returns:
        A frozenset of canonical target names (e.g. "core", "thermal", "fault"). The command
        router treats a CommandMsg whose target is outside this set as unroutable (loud NACK +
        COMMAND_UNROUTABLE fault). Derived from the dictionary so adding a command keeps the
        router's routable set in sync automatically.
    """
    return frozenset(spec.target for spec in COMMAND_DICTIONARY.values())


def hazardous_command_ids() -> frozenset[str]:
    """Return the opcode strings of every hazardous command (ARM/EXECUTE two-step).

    Returns:
        A frozenset of command_id values (e.g. "EXIT_SAFE") the router must gate behind a
        two-step ARM then EXECUTE with an inhibit re-check. Derived from the dictionary's
        hazardous flag so the router stays in sync as hazardous commands are added.
    """
    return frozenset(
        spec.command_id.value for spec in COMMAND_DICTIONARY.values() if spec.hazardous
    )


def lookup_command(command_id: str) -> Result[CommandSpec, FaultCode]:
    """Resolve a wire command_id string to its CommandSpec.

    Args:
        command_id: The opcode string from the command body.

    Returns:
        Ok(spec) if command_id names a known CommandId, else Err(FaultCode.COMMAND_INVALID).
    """
    try:
        key = CommandId(command_id)
    except ValueError:
        return Err(FaultCode.COMMAND_INVALID)
    spec = COMMAND_DICTIONARY.get(key)
    if spec is None:
        return Err(FaultCode.COMMAND_INVALID)
    return Ok(spec)


def validate_command(
    spec: CommandSpec, params: dict[str, str | int | float | bool]
) -> Result[None, FaultCode]:
    """Check params against a spec's schema: exact key set + per-key primitive kind.

    Args:
        spec: The command spec to validate against.
        params: The parameter dict from the decoded command body.

    Returns:
        Ok(None) if params exactly matches the spec's declared parameters by name and kind,
        else Err(FaultCode.COMMAND_INVALID).

    Notes:
        bool is a subclass of int in Python; this rejects a bool where INT/FLOAT is required
        and rejects int/float where BOOL is required, so kinds are enforced strictly.
    """
    expected_names = {p.name for p in spec.params}
    if set(params.keys()) != expected_names:
        return Err(FaultCode.COMMAND_INVALID)
    for param in spec.params:
        value = params[param.name]
        if isinstance(value, bool) != (param.kind is ParamKind.BOOL):
            return Err(FaultCode.COMMAND_INVALID)
        if param.kind is ParamKind.FLOAT:
            if not isinstance(value, (int, float)):
                return Err(FaultCode.COMMAND_INVALID)
        elif not isinstance(value, _KIND_TO_TYPE[param.kind]):
            return Err(FaultCode.COMMAND_INVALID)
    return Ok(None)
