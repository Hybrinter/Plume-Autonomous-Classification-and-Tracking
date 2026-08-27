# flight.iss_iface.app

**Source:** `packages/flight/src/flight/iss_iface/app.py`
**Kind:** app shell

## Purpose

`IssIfaceApp` is the station link app shell. It receives CCSDS telecommands, runs the ingress
pipeline, pumps downlink items during acquisition of signal, handles model-upload chunks, and
publishes link state and heartbeats.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `IngressState` | class | Mutable ingress state: sequence map, TM counter, upload buffer |
| `IssIfaceApp` | class | Frozen app with link, storage, bus, clock, and subscriptions |
| `IssIfaceApp.from_config` | function | Builds the app and subscribes to downlink and routed commands |
| `IssIfaceApp.pump_uplink` | method | Drains inbound packets through the ingress pipeline |
| `IssIfaceApp.pump_downlink` | method | Encodes and sends downlink items when AOS |
| `IssIfaceApp.pump_routed_commands` | method | Processes `UPLOAD_MODEL_CHUNK` routed commands |
| `IssIfaceApp.tick` | method | Publishes link state and runs all pumps once |
| `IssIfaceApp.run` | method | Periodic loop with heartbeats until `stop_event` is set |

## Inputs and outputs

- `from_config(cfg, bus, clock, link, uplink_key, storage_reader, storage_writer)` returns an
  `IssIfaceApp`.
- `pump_uplink()` returns the count of published `CommandMsg` values.
- `pump_downlink()` returns the count of sent TM packets.
- `pump_routed_commands()` returns the count of processed upload chunks.

## Behavior

1. Publish `LinkStateMsg` with the current link state from the station driver.
2. Drain inbound packets: run `process_inbound`, publish validated `CommandMsg`, always publish
   `CommandAckMsg`, publish `FaultEventMsg` on reject.
3. Drain routed `UPLOAD_MODEL_CHUNK` commands: decode base64, accumulate chunks, store the
   pair bundle, publish `ModelStagedMsg`.
4. When link state is AOS, drain `DownlinkItemMsg` values, resolve storage refs, encode CCSDS TM
   packets, and send them.
5. Emit `HeartbeatMsg` every `watchdog_interval_s`.

## Errors and faults

| FaultCode | Trigger |
| --- | --- |
| Ingress reject codes | See [`ingress/pipeline`](ingress/pipeline.md) |
| `COMM_TIMEOUT` | Station uplink receive or downlink send failure |
| `STORAGE_FULL` | Staged model store failure |
| `MODEL_CORRUPT` | Reassembled artifact CRC mismatch |
| `COMMAND_INVALID` | Bad chunk base64 or upload rejection |

Uplink receive errors stop the uplink drain early. Downlink errors emit a fault and continue.

## Messages

| Direction | Type |
| --- | --- |
| Subscribe | `DownlinkItemMsg`, `RoutedCommandMsg` |
| Publish | `CommandMsg`, `CommandAckMsg`, `FaultEventMsg`, `LinkStateMsg`, `HeartbeatMsg`, `ModelStagedMsg` |

## Configuration

| Config section | Fields used |
| --- | --- |
| `fault` | `watchdog_interval_s` (tick cadence and heartbeat interval) |
| `link` | `tm_apid` (outbound TM APID) |
| `command_ingress` | `require_auth`, `accepted_sources` |

The HMAC key is injected by the composition root, not read from config inside the app.

## Constraints

- Downlink items with `storage_ref` resolve bytes at transmission time via `StorageReader`.
- The TM sequence counter wraps at 14 bits (`0x3FFF`).
- `run()` closes the link on shutdown.
- The loop uses `stop_event.wait(timeout=...)` for immediate shutdown.

## Related documents

- [`flight.iss_iface`](../iss_iface.md)
- [`flight.iss_iface.ingress`](ingress.md)
- [`flight.iss_iface.upload`](upload.md)
- [`flight.core.downlink`](../core/downlink.md)
