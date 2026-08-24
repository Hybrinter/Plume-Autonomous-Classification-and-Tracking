# flight.iss_iface

**Source:** `packages/flight/src/flight/iss_iface`
**Kind:** subsystem app

## Purpose

The iss_iface package is the station link front door. It authenticates inbound telecommands,
publishes validated commands on the bus, reassembles model uploads, and sends downlink items
as CCSDS telemetry packets.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](iss_iface/app.md) | app shell | Uplink, downlink, and upload pumps over the station link |
| [`upload`](iss_iface/upload.md) | pure module | Chunked model artifact reassembly |
| [`ingress`](iss_iface/ingress.md) | package | Command-ingress pipeline re-exports |

## Package interface

The package re-exports `IssIfaceApp`.

## Interactions

The app uses the `StationLink`, `StorageReader`, and `StorageWriter` HAL protocols. It
subscribes to `DownlinkItemMsg` and `RoutedCommandMsg`. It publishes `CommandMsg`,
`CommandAckMsg`, `FaultEventMsg`, `LinkStateMsg`, `HeartbeatMsg`, and `ModelStagedMsg`.
Command ingress faults are log-and-continue; they never trigger SAFE mode.

## Constraints

The ingress pipeline and upload reassembly are pure modules. The app shell owns the bus,
the clock, the HMAC key, and mutable ingress state. Downlink draining runs only when the link
reports `LinkState.AOS`. An uplink receive error stops the uplink drain early to preserve
command ordering.

## Related documents

- [`flight.core.composition`](core/composition.md)
- [`flight.core.downlink`](core/downlink.md)
