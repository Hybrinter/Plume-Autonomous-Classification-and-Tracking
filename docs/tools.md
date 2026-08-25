# tools

**Source:** `packages/tools/src/tools/`
**Kind:** package

## Purpose

The tools package holds engineering utilities outside the flight image. It includes the model
acceptance gate and the SIL telemetry analysis toolchain.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`accept`](tools/accept.md) | module | Frozen ONNX artifact acceptance gate |
| [`analysis`](tools/analysis.md) | package | Deterministic SIL capture, stats, plots, and reports |

## Package interface

`tools` has no top-level `__init__.py` exports. Import from `tools.accept` or
`tools.analysis`.

Run analysis with `python -m tools.analysis run <suite|scenario> --out <dir>`.

## Interactions

`tools.accept` imports `flight.payload.model.verify` for hash and I/O contract checks.

`tools.analysis` drives `sim.sil.build_sil_system` and `step_once`, subscribes passively to
bus message types, and writes static report bundles. It never publishes to the bus or changes
flight behavior.

## Constraints

- Default tools dependencies stay on the tools package (matplotlib, pandas, pyarrow, plus
  pact-flight and pact-sim).
- Optional extras `train` and `export` are named install roles. They currently declare no
  packages.
- A flight-only install does not include `pact-tools`.
- Analysis is read-only observability over the deterministic SIL harness.
- Acceptance runs inference through an injected callable so CI stays SDK-free.

## Related documents

- [`tools.accept`](tools/accept.md)
- [`tools.analysis`](tools/analysis.md)
- [`sim.sil`](sim/sil.md)
