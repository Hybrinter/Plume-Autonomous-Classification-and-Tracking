# tools.ml_models

**Source:** `packages/tools/src/tools/ml_models/`
**Kind:** package

## Purpose

The ml_models package holds processed-pack data, network builders, the
plain-torch train loop, and ONNX export.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`data`](ml_models/data.md) | package | Packs, Zenodo reads, prism proxy, and flight canvas |
| [`arch`](ml_models/arch.md) | package | Segmentor and classifier network builders |
| [`train`](ml_models/train.md) | package | Train loop, losses, metrics, cost, and sweeps |
| [`export`](ml_models/export.md) | package | ONNX logits, acceptance, and the flight pair blob |
| [`cli`](ml_models/cli.md) | module | Typer commands for train, eval, export, accept, and pair |
| [`__main__`](ml_models/__main__.md) | module | `python -m tools.ml_models` entry shim |

## Package interface

`tools.ml_models.__init__` carries a module docstring only. Callers import
`tools.ml_models.data`, `tools.ml_models.arch`, `tools.ml_models.train`, and
`tools.ml_models.export`. Run
`pact-tools ml-models <train|eval|export|accept|pair>`
or `python -m tools.ml_models`.

## Interactions

`tools.ml_models.data.norm` calls
`flight.payload.preprocess.normalize.normalize_dn`. The package does not publish
on the bus. `tools.ml_models.data`, `tools.ml_models.arch`, and
`tools.ml_models.train` do not import `flight.payload.inference`,
`flight.core`, or `tools.analysis`. `tools.ml_models.arch` does not import
`flight`. `tools.ml_models.export` calls `flight.payload.inference.verify` and
reads `InferenceConfig`. It does not import `flight.core` or `tools.analysis`.
`tools.ml_models.train.loop` calls `arch.registry.build`. A canvas
run calls `data.canvas.sample_view`.

## Constraints

- `tools.ml_models.data` does not import torch.
- `tools.ml_models.arch` does not import `flight`.
- `tools.ml_models.train` imports torch.
- `tools.ml_models.export` imports torch.
- Pack files are local directories. This package does not fetch a corpus.
- The package `__init__` does not re-export names.

## Related documents

- [`tools`](../tools.md)
- [`tools.ml_models.data`](ml_models/data.md)
- [`tools.ml_models.arch`](ml_models/arch.md)
- [`tools.ml_models.train`](ml_models/train.md)
- [`tools.ml_models.export`](ml_models/export.md)
- [`tools.ml_models.cli`](ml_models/cli.md)
- [`flight.payload.preprocess.normalize`](../flight/payload/preprocess/normalize.md)
