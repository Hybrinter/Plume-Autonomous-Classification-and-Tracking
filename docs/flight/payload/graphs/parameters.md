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
| `default_policy` | method | Graph imaging/inference policy, enabled or fully off |
| `operating_policy` | method | `Result[EffectivePolicy, FaultCode]` for the OPERATE graph or one node |

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
5. `operating_policy(node_override=None)` resolves the OPERATE effective
   policy in inheritance order: the sensor-derived enabled base, then the
   `payload_policy.operate` table, then one named node table (`tracking`,
   `rewind`, `fast_rewind`, `hold`). Each stage validates the complete
   resolved policy and returns `Err(COMMAND_INVALID)` on an invalid
   combination; no partial policy applies. Other graphs stay fully off.
6. `encoder_fresh` requires `health.feedback_valid` and a finite encoder
   sample within `controller.integrity.feedback_max_age_s` of `now_s`.

## Errors and faults

`operating_policy` returns `Err(COMMAND_INVALID)` for an invalid resolved
combination such as enabled inference with disabled acquisition, zero duty
with enabled inference, exposure exceeding the capture interval, or a
bounds/rate violation.

## Messages

None.

## Configuration

Reads `PactConfig` through `GraphParameters`.

## Constraints

Pure: no I/O, no bus, no clock reads, no `SystemMode`.

## Related documents

- [`flight.payload.graphs`](../graphs.md)
- [`flight.payload.gimbal.request`](../gimbal/request.md)
