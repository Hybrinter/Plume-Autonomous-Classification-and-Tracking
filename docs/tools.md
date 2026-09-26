# tools

**Source:** `packages/tools/src/tools/`
**Kind:** package

## Purpose

The tools package holds engineering utilities outside the flight image. It
includes inference training, export, and acceptance under `tools.inference`,
processed-pack data, network builders, training, ONNX export, and run
analysis under `tools.ml_models`, and SIL telemetry analysis under
`tools.analysis`.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`inference`](tools/inference.md) | package | Train, export, accept, and score inference artifacts |
| [`ml_models`](tools/ml_models.md) | package | Packs, builders, training, export, and analysis |
| [`analysis`](tools/analysis.md) | package | Deterministic SIL capture, stats, plots, and reports |
| [`original_dataset_analysis`](tools/original_dataset_analysis.md) | package | Zenodo band and ground-sample-distance study |
| [`cli`](tools/cli.md) | module | Root `pact-tools` Typer application |
| [`__main__`](tools/__main__.md) | module | `python -m tools` entry shim |

## Package interface

`tools` has no top-level `__init__.py` exports. Import from `tools.inference`,
`tools.ml_models`, `tools.analysis`, or `tools.original_dataset_analysis`.

Run inference workflows with
`pact-tools inference <train|eval|report|list|compare|rank|pareto|sweep|arches|export|accept|finalize|fetch>`.

Run ml_models workflows with
`pact-tools ml-models <train|eval|export|accept|pair>`.

Run analysis with
`pact-tools analysis run <suite|scenario> --out <dir>`.

`python -m tools`, `python -m tools.inference`, `python -m tools.ml_models`,
and `python -m tools.analysis` provide module aliases.

## Interactions

`tools.ml_models.export.accept` imports `flight.payload.inference.verify` for
hash and I/O contract checks. `tools.inference.accept` re-exports that gate.

`tools.analysis` drives `sim.sil.build_sil_system` and `step_once`, subscribes
passively to bus message types, and writes static report bundles. It never
publishes to the bus or changes flight behavior.

## Constraints

- Default tools dependencies include torch and torchvision, plus Typer,
  matplotlib, pandas, pyarrow, pact-flight, and pact-sim.
- Extra `export` installs onnx and onnxruntime.
- Extra `data` installs rasterio for GeoTIFF reads during dataset preprocess.
- Workspace extra `train` installs `pact-tools[export,data]` for training boxes.
  Workspace extra `dev` includes `pact-tools` and torch. Lean CI shards sync
  extra `dev-ci-flight` and omit `pact-tools`.
- A Windows install resolves torch and torchvision from the CUDA 13.0 PyTorch
  index. A Linux or macOS install resolves both from PyPI.
- The Windows and Linux wheels bundle the CUDA 13.0 runtime. A GPU install needs
  an NVIDIA driver at CUDA 13.0 or later. It needs no CUDA toolkit.
- A flight-only install does not include `pact-tools`.
- Analysis is read-only observability over the deterministic SIL harness.
- Acceptance runs inference through an injected callable so CI stays SDK-free.

## Related documents

- [`tools.inference`](tools/inference.md)
- [`tools.ml_models`](tools/ml_models.md)
- [`tools.analysis`](tools/analysis.md)
- [`tools.original_dataset_analysis`](tools/original_dataset_analysis.md)
- [`tools.cli`](tools/cli.md)
- [`sim.sil`](sim/sil.md)
