# flight.iss_iface.ingress.pipeline

**Source:** `packages/flight/src/flight/iss_iface/ingress/pipeline.py`
**Kind:** pure module

## Purpose

The pipeline module runs the six-stage command-ingress check on one raw CCSDS telecommand packet.
It returns a validated `CommandMsg` or a rejection with fault code and ack fields.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `IngressOutcome` | class | Per-packet result: command or rejection with ack echo fields |
| `process_inbound` | function | Runs the full pipeline on one raw packet |
| `build_tc_packet` | function | Re-exported signed TC builder from `flight.libs.commands.tc` |

## Inputs and outputs

- `process_inbound(raw, key, require_auth, accepted_sources, last_seq)` returns
  `(IngressOutcome, dict[str, int])`. On accept, the returned sequence map updates the source
  entry to the command sequence number.

## Behavior

1. Decode the CCSDS frame and verify the CRC trailer.
2. Split the 32-byte HMAC tag from the JSON body.
3. Parse JSON and extract `command_id`, `params`, `source`, and `seq`.
4. Verify HMAC-SHA256 over the body bytes when `require_auth` is true.
5. Check `source` against the accepted-sources allow list.
6. Look up the command in the dictionary and validate params.
7. Reject replays when `seq` is not strictly greater than the last accepted seq for that source.
8. Build a `CommandMsg` with `target` stamped from the dictionary spec.

Wire format: CCSDS header (type 1), JSON body, 32-byte HMAC tag, CRC-32 trailer.

## Errors and faults

| FaultCode | Trigger |
| --- | --- |
| `COMMAND_CRC_FAIL` | CCSDS decode or CRC verification failure |
| `COMMAND_AUTH_FAIL` | Missing HMAC tag, HMAC mismatch, or source not in allow list |
| `COMMAND_INVALID` | Malformed JSON, unknown command, or param validation failure |
| `COMMAND_SEQ_ERROR` | Replay or duplicate sequence number |

Malformed input maps to a rejected outcome; the function does not raise.

## Messages

Produces a `CommandMsg` inside `IngressOutcome` on accept. Does not publish to the bus.

## Configuration

The caller passes `require_auth` and `accepted_sources` from `CommandIngressConfig`.

## Constraints

- Pure module with no bus, clock, or I/O access.
- The sequence map is not mutated in place; a new dict is returned on accept.
- `CommandMsg.timestamp_utc` is empty; the app shell stamps it from the clock.
- Sequence dedup is the final stage.

## Related documents

- [`flight.iss_iface.ingress`](../ingress.md)
- [`flight.iss_iface.app`](../../app.md)
- [`flight.libs.ccsds`](../../../libs/ccsds.md)
- [`flight.libs.commands`](../../../libs/commands.md)
