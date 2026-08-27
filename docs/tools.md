# tools

**Source:** `packages/tools/src/tools/`
**Kind:** package

## Purpose

The tools package holds engineering utilities outside the flight image. It includes
model training, export, and acceptance under `tools.model`, and SIL telemetry
analysis under `tools.analysis`.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`model`](tools/model.md) | package | Train, export, accept, and score frozen model artifacts |
| [`accept`](tools/accept.md) | module | Compatibility re-export of `tools.model.accept` |
| [`analysis`](tools/analysis.md) | package | Deterministic SIL capture, stats, plots, and reports |

## Package interface

`tools` has no top-level `__init__.py` exports. Import from `tools.model` or
`tools.analysis`.

Run analysis with `python -m tools.analysis run <suite|scenario> --out <dir>`.

Run model CLI with `python -m tools.model <train|export|accept>`.

Fetch the smoke-plume corpus with `python scripts/fetch_smoke_plume_dataset.py`
(checksum status only unless `--download`).

## Interactions

`tools.model.accept` imports `flight.payload.inference.verify` for hash and I/O
contract checks.

`tools.analysis` drives `sim.sil.build_sil_system` and `step_once`, subscribes
passively to bus message types, and writes static report bundles. It never
publishes to the bus or changes flight behavior.

## Constraints

- Default tools dependencies stay on the tools package (matplotlib, pandas,
  pyarrow, plus pact-flight and pact-sim).
- Optional extra `train` installs torch and torchvision. Extra `export` installs
  onnx.
- A flight-only install does not include `pact-tools`.
- Analysis is read-only observability over the deterministic SIL harness.
- Acceptance runs inference through an injected callable so CI stays SDK-free.

## Related documents

- [`tools.model`](tools/model.md)
- [`tools.accept`](tools/accept.md)
- [`tools.analysis`](tools/analysis.md)
- [`sim.sil`](sim/sil.md)
