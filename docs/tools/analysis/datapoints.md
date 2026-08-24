# tools.analysis.datapoints

**Source:** `packages/tools/src/tools/analysis/datapoints.py`
**Kind:** pure module

## Purpose

Datapoints defines the typed per-step signal registry for SIL capture. Every observable
sampled each step is one frozen `Signal` with a pure extractor over `SampleContext`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SignalKind` | enum | `NUMERIC` or `CATEGORICAL` |
| `Signal` | class | Name, group, title, unit, kind, extractor callable |
| `DeviceSample` | class | One-shot sim HAL driver snapshot |
| `SampleContext` | class | Per-step inputs for extractors |
| `MESSAGE_TYPES` | constant | Nineteen bus message types in stable order |
| `MONITORED` | constant | Nine FDIR-monitored subsystem names |
| `REGISTRY` | constant | Full assembled signal tuple |
| `GROUPS` | constant | Ordered de-duplicated group names |
| `signals_for_group` | function | Signals in one group |
| `signal_names` | function | All signal names in registry order |
| `is_event_rate` | function | True for per-step count signals |
| `accumulable_names` | function | Names that get `.cumulative` columns |
| `is_nan` | function | True for float NaN sentinel values |

## Inputs and outputs

Extractors map `SampleContext -> float | str`. The recorder evaluates all registry signals
each step.

## Behavior

1. `build_registry` assembles signals from per-group builders: system, bus, payload, fault,
   iss_iface, thermal, electrical, command_router, storage, downlink, mechanical,
   model_deploy, and enrichment.
2. Bus signals emit publish count, queue depth, drops, and overflow per message type.
3. Payload signals read control state, Kalman and EMA estimators, gimbal driver samples, and
   bus output counts.
4. Fault signals read latch state, watchdog misses, heartbeat counts, and per-code fault
   rates.
5. Duplicate signal names raise `ValueError` at registry build time.

## Errors and faults

Individual extractor exceptions are handled by the recorder, not this module.

## Messages

Extractors read drained copies from the recorder subscriptions. They do not subscribe.

## Configuration

None.

## Constraints

- Extractors are read-only and side-effect-free.
- No getattr-style dynamic dispatch. Each signal carries a typed `ExtractorFn`.
- Cumulative columns apply only to per-step event counts (titles ending in "/step" or
  "this step").
- `read_position` is sampled once per step in `DeviceSample` for consistency.

## Related documents

- [`tools.analysis`](analysis.md)
- [`tools.analysis.recorder`](recorder.md)
- [`tools.analysis.stats`](stats.md)
