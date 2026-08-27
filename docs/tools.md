# tools

**Source:** `packages/tools/src/tools/`
**Kind:** package

## Purpose

The tools package holds engineering utilities outside the flight image. It
includes inference training, export, and acceptance under `tools.inference`,
and SIL telemetry analysis under `tools.analysis`.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`inference`](tools/inference.md) | package | Train, export, accept, and score inference artifacts |
| [`analysis`](tools/analysis.md) | package | Deterministic SIL capture, stats, plots, and reports |
| [`cli`](tools/cli.md) | module | Root `pact-tools` Typer application |
| [`__main__`](tools/__main__.md) | module | `python -m tools` entry shim |

## Package interface

`tools` has no top-level `__init__.py` exports. Import from `tools.inference` or
`tools.analysis`.

Run inference workflows with
`pact-tools inference <train|export|accept|fetch>`.

Run analysis with
`pact-tools analysis run <suite|scenario> --out <dir>`.

`python -m tools`, `python -m tools.inference`, and
`python -m tools.analysis` provide module aliases.

## Interactions

`tools.inference.accept` imports `flight.payload.inference.verify` for hash and I/O
contract checks.

`tools.analysis` drives `sim.sil.build_sil_system` and `step_once`, subscribes
passively to bus message types, and writes static report bundles. It never
publishes to the bus or changes flight behavior.

## Constraints

- Default tools dependencies stay on the tools package (Typer, matplotlib,
  pandas, pyarrow, plus pact-flight and pact-sim).
- Optional extra `train` installs torch and torchvision. Extra `export` installs
  onnx.
- A flight-only install does not include `pact-tools`.
- Analysis is read-only observability over the deterministic SIL harness.
- Acceptance runs inference through an injected callable so CI stays SDK-free.

## Related documents

- [`tools.inference`](tools/inference.md)
- [`tools.analysis`](tools/analysis.md)
- [`tools.cli`](tools/cli.md)
- [`sim.sil`](sim/sil.md)
