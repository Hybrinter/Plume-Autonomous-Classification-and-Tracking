# flight.payload.graphs.parameters

**Source:** `packages/flight/src/flight/payload/graphs/parameters.py`
**Kind:** pure module

## Purpose

`GraphParameters` is the single typed config projection handed to every pure
graph and node function. It derives envelopes, camera geometry, plant terms,
policy defaults, and sensor limits from `PactConfig` without tuning constants
or touching hardware.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `GraphParameters` | dataclass | `PactConfig` projection plus `detailed_plant` and `policy_revision` |
| `encoder_fresh` | function | Freshness check: valid health plus finite, timely encoder sample |

## Inputs and outputs

`GraphParameters` projects `PactConfig` once. `encoder_fresh(inputs, params) -> bool`
judges a `TickInputs` encoder sample against the health flag, the configured
maximum age, and the hardware angle range.

## Behavior

1. `science_envelope`/`hardware_envelope` bound the science window and
   hardware travel under the hardware slew cap.
2. `pose_envelope` caps the rate at `position.r_max` or the hardware slew,
   whichever is smaller; `stow_envelope` caps at the bounded stow rate.
3. `max_decel_rad_s2`/`rate_loop_bandwidth_rad_s` are finite with the detailed
   plant and unbounded (inf) otherwise.
4. `default_policy(enabled)` builds the graph imaging/inference policy;
   inference runs only when enabled and the duty cycle is nonzero.
5. `encoder_fresh` requires `health.feedback_valid` and a finite encoder
   sample within `controller.integrity.feedback_max_age_s` of `now_s`.

## Errors and faults

None.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure: no I/O, no bus, no clock reads, no `SystemMode`.

## Related documents

- [`flight.payload.graphs`](../graphs.md)
- [`flight.payload.gimbal.request`](../gimbal/request.md)
