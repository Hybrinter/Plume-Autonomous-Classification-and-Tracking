# flight.iss_iface.ingress

**Source:** `packages/flight/src/flight/iss_iface/ingress`
**Kind:** package

## Purpose

This package holds the pure command-ingress pipeline. It validates raw CCSDS telecommand
bytes and returns either a `CommandMsg` or a rejection outcome.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`pipeline`](ingress/pipeline.md) | pure module | Full ingress pipeline from raw bytes to outcome |

## Package interface

The package re-exports `IngressOutcome`, `process_inbound`, and `build_tc_packet`.

## Interactions

None. The app shell calls `process_inbound` and publishes the results on the bus.

## Constraints

All ingress logic lives in the pipeline module. The pipeline is pure and stateless except for
the threaded per-source sequence map.

## Related documents

- [`flight.iss_iface`](../iss_iface.md)
- [`flight.iss_iface.app`](../app.md)
