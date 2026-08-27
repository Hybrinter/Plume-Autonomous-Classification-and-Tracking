# tools.inference

**Source:** `packages/tools/src/tools/inference/`
**Kind:** package

## Purpose

The inference package holds classifier and segmentor training, ONNX export,
artifact acceptance, dataset preparation, and model metrics.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`accept`](inference/accept.md) | module | Frozen ONNX intake gate |
| [`metrics`](inference/metrics.md) | module | Classifier and segmentor design metrics |
| [`data`](inference/data.md) | module | Synthetic scenes, processed packs, and torch Dataset loaders |
| [`split`](inference/split.md) | module | Frozen train/val/test recipe and dataset hash |
| [`fetch`](inference/fetch.md) | module | Zenodo 4250706 fetch, unpack, labeled preprocess |
| [`train`](inference/train.md) | module | Plain-torch SGD train loop |
| [`export`](inference/export.md) | module | ONNX export, manifest, and promote |
| [`arch`](inference/arch.md) | package | U-Net segmentor and ResNet-50 classifier |
| [`cli`](inference/cli.md) | module | Typer commands for inference workflows |
| [`__main__`](inference/__main__.md) | module | `python -m tools.inference` entry shim |

## Package interface

Re-exports: `Manifest`, `GoldenScene`, `GoldenClassifierScene`,
`AcceptanceReport`, `ClassifierAcceptanceReport`, `load_manifest`,
`compute_iou`, `accept_artifact`, `accept_classifier_artifact`,
`onnx_inference_fn`, `onnx_classifier_inference_fn`, `binary_accuracy`,
`mean_binary_accuracy`.

Run `pact-tools inference <train|export|accept|fetch>` or
`python -m tools.inference <train|export|accept|fetch>`.

## Interactions

`tools.inference.accept` imports `flight.payload.inference.verify`. `tools.inference.data`
imports `flight.payload.preprocess.normalize_dn`. The package does not drive
the SIL and does not publish on the bus.

## Constraints

- Default `pact-tools` install includes torch and torchvision. The `export`
  extra adds onnx.
- Flight must not import `tools`.
- The smoke-plume corpus stays out of git. Run
`python scripts/fetch_smoke_plume_dataset.py` for checksum status. Pass
`--download` only on a training box. Pass `--preprocess` to write a labeled
processed pack with frozen splits.

## Related documents

- [`tools`](../tools.md)
- [`tools.analysis`](analysis.md)
- [`flight.payload.inference.verify`](../flight/payload/inference/verify.md)
