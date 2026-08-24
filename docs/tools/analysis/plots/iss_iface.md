# tools.analysis.plots.iss_iface

**Source:** `packages/tools/src/tools/analysis/plots/iss_iface.py`
**Kind:** module

## Purpose

The iss_iface plot builder renders station link state, downlink traffic, ingress replay guard,
upload reassembly, and per-step message flow.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build` | function | Build iss_iface figures from a wide DataFrame |

## Behavior

1. Station link state categorical timeline.
2. Outbound TM sequence and cumulative station packets sent.
3. Known sources, upload buffer, total chunks, and upload progress fraction.
4. Stacked per-step command, ack, link-state, model-staged, and upload-chunk counts.
5. Cumulative commands published and acks.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

None.

## Related documents

- [`tools.analysis.plots`](plots.md)
- [`tools.analysis.plots.common`](common.md)
