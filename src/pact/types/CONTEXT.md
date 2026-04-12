# types/ -- Agent Context

## Purpose

Dependency root. Defines every cross-boundary type: message dataclasses, enums, config
dataclasses, and `Result[T,E]`. No business logic. The only package all other packages
may import from.

## Defining Design Decision

`DownlinkPriority` is `enum.Enum` not `IntEnum`. Its integer values (0-3) are used
internally by `queue.PriorityQueue`, but must never be serialized as bare ints in CCSDS
packets. When passing to a priority queue, use `.value` explicitly:
`downlink_queue.put((item.priority.value, item))`.

## Invariants

- `numpy` arrays in frozen dataclasses (e.g., `RawFrameMsg.raw_bands`) are typed as
  `object` at runtime -- frozen dataclasses cannot enforce numpy dtype or shape at
  construction time. Validate shape in the producing subsystem before queuing.
- Default field values in `config.py` must match `config/default.toml` exactly.
  There is no CI check; divergence silently breaks test reproducibility.

## Gotchas

No schema versioning exists. If message schemas change between software versions during
a multi-day mission, queued messages may be structurally incompatible. Add
`schema_version: int` to `RawFrameMsg` and `InferenceResultMsg` before the first flight
integration test.

## Phase II Gaps

None structural. Schema versioning is the only pending work.
