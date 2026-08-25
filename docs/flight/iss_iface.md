# flight.iss_iface

**Source:** `packages/flight/src/flight/iss_iface`
**Kind:** subsystem app

## Purpose

The iss_iface package is the payload seam onto the station-owned link. It authenticates inbound
telecommands, publishes validated commands on the bus, encodes downlink items as CCSDS TM packets,
and reassembles chunked model uploads.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](iss_iface/app.md) | module | Station link app shell: uplink, downlink, upload, heartbeats |
| [`ingress`](iss_iface/ingress.md) | package | Pure command-ingress pipeline |
| [`upload`](iss_iface/upload.md) | module | Chunked model-upload reassembly buffer |

## Package interface

Re-exports `IssIfaceApp`.

## Interactions

Uses the `StationLink`, `StorageReader`, and `StorageWriter` HAL protocols. Subscribes to
`DownlinkItemMsg` and `RoutedCommandMsg`. Publishes `CommandMsg`, `CommandAckMsg`, `FaultEventMsg`,
`LinkStateMsg`, `HeartbeatMsg`, and `ModelStagedMsg`. The composition root injects the HMAC key
bytes; the app does not read the key file.

## Constraints

- The ingress pipeline is pure; the app shell owns the bus, clock, HMAC key, and mutable state.
- Downlink drains only when the link reports `LinkState.AOS`.
- Every inbound packet produces exactly one `CommandAckMsg`.
- Command ingress faults are log-and-continue; they do not trigger SAFE mode.
- `CommandMsg.target` comes from the command dictionary, not from the ground frame.

## Related documents

- [`flight.core.downlink`](core/downlink.md)
- [`flight.core.model_deploy`](core/model_deploy.md)
- [`flight.fault`](fault.md)
- [`flight.libs.ccsds`](libs/ccsds.md)
- [`flight.libs.commands`](libs/commands.md)
