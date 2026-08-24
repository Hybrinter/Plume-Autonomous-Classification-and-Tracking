# flight.libs

**Source:** `packages/flight/src/flight/libs`
**Kind:** package

## Purpose

`flight.libs` holds shared flight libraries. It supplies the message bus, message
dataclasses, enumerations, configuration types, CCSDS codec, command dictionary, clock
abstraction, structured logging, and version accessor. Subsystems import from package
roots such as `flight.libs.types` and `flight.libs.messages`.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`version`](flight/libs/version.md) | module | Flight software version string |
| [`bus`](flight/libs/bus.md) | package | Typed in-process pub/sub bus |
| [`ccsds`](flight/libs/ccsds.md) | package | CCSDS Space Packet encode/decode |
| [`commands`](flight/libs/commands.md) | package | Command dictionary and TC builder |
| [`config`](flight/libs/config.md) | package | Frozen configuration dataclasses |
| [`messages`](flight/libs/messages.md) | package | Bus message dataclasses |
| [`telemetry`](flight/libs/telemetry.md) | package | Structured logging setup |
| [`time`](flight/libs/time.md) | package | Injectable clock abstraction |
| [`types`](flight/libs/types.md) | package | Enumerations, `Result`, frame types |

## Package interface

None. The directory has no top-level `__init__.py`. Import from subpackage roots
(`flight.libs.bus`, `flight.libs.config`, and the others listed above).

## Interactions

- `flight.libs.bus` delivers message dataclasses from `flight.libs.messages`.
- `flight.libs.messages` imports enumerations from `flight.libs.types`.
- `flight.libs.commands` validates commands with types from `flight.libs.types`.
- `flight.libs.ccsds` returns `Result[..., FaultCode]` from `flight.libs.types`.
- `flight.libs.config` defines typed config consumed by subsystem apps and the composition
  root.
- `flight.libs.time` supplies the `Clock` protocol injected by composition roots.
- `flight.libs.telemetry` configures structlog at process startup.

## Constraints

- `flight.libs` is the bottom import layer. No `flight.libs` module imports subsystem apps.
- Import from package roots, not inner modules (for example `flight.libs.types`, not
  `flight.libs.types.enums`).
- Within `libs`, `types` sits below `messages`. `config`, `bus`, `time`, `telemetry`,
  `ccsds`, and `commands` do not depend on each other except where noted (`commands`
  imports `types`; `ccsds` imports `types`).

## Related documents

- [`flight`](flight.md)
- [`flight.core`](flight/core.md)
