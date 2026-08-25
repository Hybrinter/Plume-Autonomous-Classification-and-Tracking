# flight.iss_iface.ingress

**Source:** `packages/flight/src/flight/iss_iface/ingress`
**Kind:** package

## Purpose

The ingress package holds the pure command-ingress pipeline. It converts raw CCSDS telecommand bytes
into a validated `CommandMsg` or a structured rejection outcome.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`pipeline`](ingress/pipeline.md) | module | Six-stage ingress pipeline and `IngressOutcome` type |

## Package interface

Re-exports `IngressOutcome`, `build_tc_packet`, and `process_inbound`.

## Interactions

None at the package level. The pipeline calls `flight.libs.ccsds` and `flight.libs.commands`. The
app shell converts outcomes into bus messages.

## Constraints

- All pipeline functions are pure: no bus, clock, or I/O.
- State threads in and out via the per-source sequence map.
- `build_tc_packet` lives in `flight.libs.commands.tc` and is re-exported here for compatibility.

## Related documents

- [`flight.iss_iface`](../iss_iface.md)
- [`flight.iss_iface.app`](../app.md)
- [`flight.iss_iface.ingress.pipeline`](ingress/pipeline.md)
- [`flight.libs.ccsds`](../../libs/ccsds.md)
- [`flight.libs.commands`](../../libs/commands.md)
