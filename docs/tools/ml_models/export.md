# tools.ml_models.export

**Source:** `packages/tools/src/tools/ml_models/export/`
**Kind:** package

## Purpose

The export package writes ONNX logits, runs the acceptance gate, and builds
the classifier plus segmentor pair blob.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`onnx`](export/onnx.md) | module | ONNX export, quantization, and manifests |
| [`accept`](export/accept.md) | module | Hash, I/O contract, and golden-scene gate |
| [`finalize`](export/finalize.md) | module | Test eval, export, and acceptance for one run |
| [`ort_providers`](export/ort_providers.md) | module | onnxruntime provider preference |
| [`pair`](export/pair.md) | module | Flight-promotable check and the pair blob |

## Package interface

`tools.ml_models.export.__init__` carries a module docstring only. Callers
import each module by name.

## Interactions

`onnx` and `accept` call `flight.payload.inference.verify`. `pair` and
`finalize` read `InferenceConfig`. `finalize` calls `tools.inference.eval`.
No module imports `flight.core` or `tools.analysis`. No module publishes on
the bus.

## Constraints

- The package imports torch.
- `flight.payload.inference.verify` supplies the hash and the I/O check.
- A flight-promotable run traces the `InferenceConfig` frame.
- A research run traces the checkpoint height and width.
- The pair blob is written for a flight-promotable run whose sidecars match
  that frame.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.export.onnx`](export/onnx.md)
- [`tools.ml_models.export.accept`](export/accept.md)
- [`tools.ml_models.export.finalize`](export/finalize.md)
- [`tools.ml_models.export.ort_providers`](export/ort_providers.md)
- [`tools.ml_models.export.pair`](export/pair.md)
- [`tools.ml_models.cli`](cli.md)
- [`flight.payload.inference.verify`](../../flight/payload/inference/verify.md)
