# flight.iss_iface.app

**Source:** `packages/flight/src/flight/iss_iface/app.py`
**Kind:** app shell

## Purpose

`IssIfaceApp` bridges the station link and the message bus. It receives CCSDS telecommands,
runs the ingress pipeline, publishes validated commands, sends downlink items as telemetry
packets, and reassembles chunked model uploads.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `IngressState` | class | Mutable per-source sequence map, TM sequence counter, and upload buffer |
| `IssIfaceApp` | class | Frozen holder of link, storage, bus, clock, config, and subscriptions |
| `IssIfaceApp.from_config` | function | Builds the app and subscribes to downlink items and routed commands |
| `IssIfaceApp.pump_uplink` | method | Drains inbound packets and publishes commands or faults |
| `IssIfaceApp.pump_downlink` | method | Encodes and sends downlink items when the link is AOS |
| `IssIfaceApp.pump_routed_commands` | method | Processes routed model-upload chunks |
| `IssIfaceApp.tick` | method | Publishes link state and runs all pumps once |
| `IssIfaceApp.run` | method | Periodic loop with heartbeats until the stop event is set |

## Inputs and outputs

`from_config(cfg, bus, clock, link, uplink_key, storage_reader, storage_writer)` returns an
`IssIfaceApp`.

`pump_uplink()` returns the count of published `CommandMsg` values.

`pump_downlink()` returns the count of sent telemetry packets.

`pump_routed_commands()` returns the count of processed upload chunks.

## Behavior

1. Publish `LinkStateMsg` with the current link state from the driver.
2. Drain inbound packets via `receive_packet`. Run each through `process_inbound`. Stamp and
   publish accepted commands. Publish a `CommandAckMsg` for every packet. Publish a
   `FaultEventMsg` on reject. Stop the drain on a link receive error.
3. Drain routed `UPLOAD_MODEL_CHUNK` commands. Decode base64 chunk data, accumulate via
   `add_chunk`, store the completed artifact, and publish `ModelStagedMsg`.
4. When the link is AOS, drain `DownlinkItemMsg` values. Resolve `storage_ref` items through
   the storage reader. Encode each body as a CCSDS TM packet and send it.

## Errors and faults

The app publishes `FaultEventMsg` for link errors, ingress rejects, storage failures, encode
failures, and send failures. It does not return `Err` from public methods.

| Fault code | Trigger |
| --- | --- |
| From ingress pipeline | CRC, auth, sequence, or validation failure |
| From storage | Downlink read failure or staged model store failure |
| From link | Uplink receive failure or downlink send failure |
| `COMMAND_INVALID` | Bad base64 in an upload chunk |
| `MODEL_CORRUPT` | Reassembled artifact CRC mismatch |

## Messages

| Direction | Type |
| --- | --- |
| Subscribe | `DownlinkItemMsg`, `RoutedCommandMsg` |
| Publish | `CommandMsg`, `CommandAckMsg`, `FaultEventMsg`, `LinkStateMsg`, `HeartbeatMsg`, `ModelStagedMsg` |

## Configuration

| Field | Source | Use |
| --- | --- | --- |
| `watchdog_interval_s` | `FaultConfig` | Tick interval and heartbeat pacing in `run` |
| `tm_apid` | `LinkConfig` | APID for outbound telemetry packets |
| `require_auth` | `CommandIngressConfig` | HMAC verification toggle |
| `accepted_sources` | `CommandIngressConfig` | Command origin allow-list |

The HMAC key is injected at construction; the app does not read it from disk.

## Constraints

`IssIfaceApp` is frozen. `IngressState` holds mutable sequence and upload state. The app
closes the link when `run` exits. Command acks flow through the downlink manager; this app
does not downlink acks directly.

## Related documents

- [`flight.iss_iface`](../iss_iface.md)
- [`flight.iss_iface.upload`](upload.md)
- [`flight.iss_iface.ingress`](ingress.md)
