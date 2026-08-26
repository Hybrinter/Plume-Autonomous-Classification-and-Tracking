# tools.model

**Source:** `packages/tools/src/tools/model/`
**Kind:** package

## Purpose

The model package holds model training, ONNX export, artifact acceptance, and
model metrics. It is the single model-engineering package under tools.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`accept`](model/accept.md) | module | Frozen ONNX intake gate |
| [`metrics`](model/metrics.md) | module | Classifier accuracy helpers |
| [`train`](model/train.md) | module | Train-loop scaffold (stub) |
| [`export`](model/export.md) | module | ONNX-export scaffold (stub) |
| [`__main__`](model/__main__.md) | module | `python -m tools.model` entry |

## Package interface

Re-exports: `Manifest`, `GoldenScene`, `AcceptanceReport`, `load_manifest`,
`compute_iou`, `accept_artifact`, `onnx_inference_fn`, `binary_accuracy`,
`mean_binary_accuracy`.

Run `python -m tools.model <train|export|accept>`. Train, export, and accept CLI
subcommands exit 2 in this layer. Call `accept_artifact` from Python.

## Interactions

`tools.model.accept` imports `flight.payload.inference.verify`. The package does not
drive the SIL and does not publish on the bus.

## Constraints

- Default `pact-tools` install does not pull torch. The `train` extra adds torch
  and torchvision. The `export` extra adds onnx.
- `import tools.model` does not import torch.
- `train` and `export` modules are stubs in this layer.
- Flight must not import `tools`.

## Related documents

- [`tools`](../tools.md)
- [`tools.analysis`](analysis.md)
- [`flight.payload.inference.verify`](../flight/payload/inference/verify.md)
