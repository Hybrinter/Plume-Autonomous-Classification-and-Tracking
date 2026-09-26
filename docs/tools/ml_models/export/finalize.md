# tools.ml_models.export.finalize

**Source:** `packages/tools/src/tools/ml_models/export/finalize.py`
**Kind:** module

## Purpose

This module scores the test split of a trained run, exports FP32 and INT8 ONNX
artifacts, and runs the golden-scene acceptance gate. It writes `finalize.json`
into the run directory.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `FinalizeReport` | class | Paths, gate outcomes, and optional promote destination |
| `finalize` | function | Evaluate test, export, accept, and write `finalize.json` |

## Inputs and outputs

`finalize(run_dir, *, int8=True, calib_samples=32, scenes_limit=0, min_iou=0.5,
min_accuracy=0.9, max_latency_ms=20.0, promote_path=None, flight=False) ->
FinalizeReport`.

The report is also written as `run_dir/finalize.json`. Export artifacts land
under `run_dir/export/`.

## Behavior

1. Load `config.toml` and locate the processed pack.
2. Score `checkpoints/best.pt` on the test split.
3. Export an FP32 ONNX graph. Export INT8 as well when `int8` is true.
   A flight-promotable run traces the `InferenceConfig` frame.
4. Run the golden-scene gate. With `flight` false, expected shapes come from
   the exported manifest. With `flight` true, expected shapes come from
   `InferenceConfig`.
5. Copy the preferred accepted artifact when `promote_path` is set. INT8 is
   preferred when it passed. FP32 is used when INT8 did not pass.
6. Write `finalize.json`.

## Errors and faults

`FileNotFoundError` when the run, checkpoint, or pack is missing.
`ImportError` when onnxruntime is missing. `ValueError` from export or from a
manifest shape that is not concrete NCHW.

## Messages

None.

## Configuration

Callers pass IoU, accuracy, latency, calibration count, scene limit, and
`flight`. Default `max_latency_ms` is `FaultConfig.inference_timeout_ms`.

## Constraints

The live gate needs onnxruntime. INT8 calibration reads the train split of the
same pack the run trained on. `flight` false does not force the flight frame
on the gate.

## Related documents

- [`tools.ml_models.export`](../export.md)
- [`tools.ml_models.export.onnx`](onnx.md)
- [`tools.ml_models.export.accept`](accept.md)
- [`tools.inference.eval`](../../inference/eval.md)
- [`tools.inference.finalize`](../../inference/finalize.md)
