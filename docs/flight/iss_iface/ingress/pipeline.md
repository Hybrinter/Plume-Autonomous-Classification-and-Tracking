# flight.iss_iface.ingress.pipeline

**Source:** `packages/flight/src/flight/iss_iface/ingress/pipeline.py`
**Kind:** pure module

## Purpose

This module runs one inbound CCSDS telecommand packet through decode, authentication,
validation, and sequence checks. It returns an `IngressOutcome` and an updated sequence map.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `IngressOutcome` | class | Per-packet result with command, ack status, and echo fields |
| `process_inbound` | function | Runs the full ingress pipeline on one raw packet |
| `build_tc_packet` | function | Re-export of the signed TC builder from `flight.libs.commands` |

## Inputs and outputs

`process_inbound(raw, key, require_auth, accepted_sources, last_seq)` returns
`(outcome, new_last_seq)`.

On accept, `outcome.command` holds a `CommandMsg` with `target` from the command dictionary.
On reject, `outcome.command` is `None` and `outcome.status` is `REJECTED`.

## Behavior

1. Decode the CCSDS frame and verify the CRC-32 trailer.
2. Split the 32-byte HMAC tag from the body. Parse the body as JSON with `command_id`,
   `params`, `source`, and `seq`.
3. When `require_auth` is true, verify HMAC-SHA256 over the JSON body bytes.
4. Check that `source` is in the accepted-sources list.
5. Look up the command in the dictionary and validate params against the spec.
6. Reject when `seq` is less than or equal to the last accepted seq for that source.
7. Build a `CommandMsg` with `target` from the dictionary. Update `new_last_seq[source]`.

Wire format: CCSDS header (type 1), JSON body, 32-byte HMAC tag, CRC-32 trailer.

## Errors and faults

| Fault code | Trigger |
| --- | --- |
| `COMMAND_CRC_FAIL` | CCSDS decode or CRC failure |
| `COMMAND_AUTH_FAIL` | Missing HMAC tag, HMAC mismatch, or source not accepted |
| `COMMAND_INVALID` | Malformed JSON, unknown command, or param validation failure |
| `COMMAND_SEQ_ERROR` | Replay or duplicate sequence number |

## Messages

None. The caller publishes `CommandMsg` and `CommandAckMsg` from the outcome.

## Configuration

None. The caller passes `require_auth`, `accepted_sources`, and the HMAC key as arguments.

## Constraints

The module is pure. It performs no I/O, reads no clock, and touches no bus. Malformed input
maps to a rejected outcome; the module does not raise. `CommandMsg.timestamp_utc` is empty;
the app shell stamps it with the clock.

## Related documents

- [`flight.iss_iface.ingress`](../ingress.md)
- [`flight.iss_iface.app`](../app.md)
