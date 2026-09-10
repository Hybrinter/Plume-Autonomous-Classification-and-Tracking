# flight.libs

**Source:** `packages/flight/src/flight/libs/`
**Kind:** package

## Purpose

The libs package holds shared flight primitives. Subsystems import types, messages, config,
bus, CCSDS, commands, clock, and logging from here. Nothing in libs imports subsystem apps.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`types`](libs/types.md) | package | Enumerations, `Result`, and raw-frame value types |
| [`messages`](libs/messages.md) | package | Frozen bus message dataclasses |
| [`config`](libs/config.md) | package | Frozen per-subsystem configuration dataclasses |
| [`bus`](libs/bus.md) | package | Typed in-process pub/sub message bus |
| [`ccsds`](libs/ccsds.md) | package | CCSDS Space Packet codec with CRC-32 trailer |
| [`commands`](libs/commands.md) | package | Typed command dictionary and TC packet builder |
| [`time`](libs/time.md) | package | Injectable clock abstraction |
| [`telemetry`](libs/telemetry.md) | package | Structured logging configuration |
| [`version`](libs/version.md) | module | Flight package version string |

## Package interface

`flight.libs.__init__` is empty. Import from package roots such as `flight.libs.types` and
`flight.libs.messages`. Do not import inner submodules directly.

## Interactions

Libs code is pure library code. It does not publish or subscribe on the bus. Subsystems and
HAL drivers import libs types. The composition root passes config dataclasses into apps.
`flight.libs.ccsds` and `flight.libs.commands` are shared by `iss_iface` and station drivers.

Layer order within libs: `types` sits below `messages`. `config`, `bus`, `time`, `telemetry`,
and `ccsds` are mutually independent. `commands` imports `types` for `CommandId`, `ParamKind`,
`Result`, and `FaultCode`.

## Constraints

- Import from package roots only. Inner submodule paths are not a public contract.
- Default config field values must match `config/default.toml` exactly.
- Library functions return `Result`. They do not raise for caller-handled failures.
- `MosaicFrame` and preprocessing tensors never ride the bus.
- Enum string values equal member names, except `DownlinkPriority` uses ints `0..3`.
- `Clock` separates monotonic time, UTC seconds, and wall-clock ISO stamps. Pure logic
  receives time as arguments; it does not read a clock.

## Related documents

- [`flight`](../flight.md)
- [`flight.libs.types`](libs/types.md)
- [`flight.libs.messages`](libs/messages.md)
