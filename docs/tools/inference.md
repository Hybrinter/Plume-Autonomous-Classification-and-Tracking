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
| [`annotations`](inference/annotations.md) | module | Polygon segmentation labels for the Zenodo corpus |
| [`metrics`](inference/metrics.md) | module | Torch classifier and segmentor design metrics |
| [`data`](inference/data.md) | module | Synthetic scenes, processed packs, and torch Dataset loaders |
| [`split`](inference/split.md) | module | Frozen train/val/test recipe and dataset hash |
| [`fetch`](inference/fetch.md) | module | Zenodo 4250706 fetch, unpack, labeled preprocess |
| [`losses`](inference/losses.md) | module | Training objectives for classifier and segmentor |
| [`train`](inference/train.md) | module | Plain-torch train loop and local run directories |
| [`cost`](inference/cost.md) | module | Parameter and FLOP counts for a logits graph |
| [`sweep`](inference/sweep.md) | module | Cartesian search space, val eval, and JSONL |
| [`eval`](inference/eval.md) | module | Held-out split scoring for a run |
| [`plots`](inference/plots.md) | module | Headless curves, overlays, and failure gallery |
| [`report`](inference/report.md) | module | Figures and markdown into a run directory |
| [`runs`](inference/runs.md) | module | List and compare local run directories |
| [`pareto`](inference/pareto.md) | module | Size versus quality frontier over the local run catalog |
| [`export`](inference/export.md) | module | ONNX export, optional INT8 PTQ, manifest, and promote |
| [`finalize`](inference/finalize.md) | module | Test eval, ONNX export, and acceptance for a trained run |
| [`arch`](inference/arch.md) | package | Architecture builders and registry grammar |
| [`cli`](inference/cli.md) | module | Typer commands for inference workflows |
| [`__main__`](inference/__main__.md) | module | `python -m tools.inference` entry shim |

## Package interface

Re-exports: `Manifest`, `GoldenScene`, `GoldenClassifierScene`,
`AcceptanceReport`, `ClassifierAcceptanceReport`, `load_manifest`,
`compute_iou`, `accept_artifact`, `accept_classifier_artifact`,
`onnx_inference_fn`, `onnx_classifier_inference_fn`, `binary_accuracy`,
`mean_binary_accuracy`.

Run `pact-tools inference <train|eval|report|list|compare|rank|pareto|sweep|arches|export|accept|finalize|fetch>`
or `python -m tools.inference <train|eval|report|list|compare|rank|pareto|sweep|arches|export|accept|finalize|fetch>`.

## Interactions

`tools.inference.accept` imports `flight.payload.inference.verify`. `tools.inference.data`
imports `flight.payload.preprocess.normalize_dn`. The package does not drive
the SIL and does not publish on the bus.

## Constraints

- Default `pact-tools` install includes torch and torchvision. The `export`
  extra adds onnx and onnxruntime. The `data` extra adds rasterio for GeoTIFF
  reads during preprocess. Workspace extra `train` installs both.
- A Windows install resolves torch and torchvision from the CUDA 13.0 PyTorch
  index. Train and eval then reach an NVIDIA GPU from the default install.
- Flight must not import `tools`.
- The smoke-plume corpus stays out of git. Run
`python scripts/fetch_smoke_plume_dataset.py` for checksum status. Pass
`--download` only on a training box. Pass `--preprocess` to write a labeled
processed pack with frozen splits.

## Related documents

- [`tools`](../tools.md)
- [`tools.analysis`](analysis.md)
- [`flight.payload.inference.verify`](../flight/payload/inference/verify.md)
