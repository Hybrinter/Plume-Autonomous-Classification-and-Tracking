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
| [`data`](model/data.md) | module | Synthetic and on-disk training batches |
| [`train`](model/train.md) | module | Plain-torch SGD train loop |
| [`export`](model/export.md) | module | ONNX export, manifest, and promote |
| [`arch`](model/arch.md) | package | U-Net segmentor and ResNet-50 classifier |
| [`__main__`](model/__main__.md) | module | `python -m tools.model` entry |

## Package interface

Re-exports: `Manifest`, `GoldenScene`, `GoldenClassifierScene`,
`AcceptanceReport`, `ClassifierAcceptanceReport`, `load_manifest`,
`compute_iou`, `accept_artifact`, `accept_classifier_artifact`,
`onnx_inference_fn`, `onnx_classifier_inference_fn`, `binary_accuracy`,
`mean_binary_accuracy`.

Run `python -m tools.model <train|export|accept>`.

## Interactions

`tools.model.accept` imports `flight.payload.inference.verify`. `tools.model.data`
imports `flight.payload.preprocess.normalize_dn`. The package does not drive
the SIL and does not publish on the bus.

## Constraints

- Default `pact-tools` install does not pull torch. The `train` extra adds torch
  and torchvision. The `export` extra adds onnx.
- `import tools.model` does not import torch.
- Flight must not import `tools`.

## Related documents

- [`tools`](../tools.md)
- [`tools.analysis`](analysis.md)
- [`flight.payload.inference.verify`](../flight/payload/inference/verify.md)
