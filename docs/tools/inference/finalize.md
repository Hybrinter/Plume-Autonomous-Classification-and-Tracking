# tools.inference.finalize

**Source:** `packages/tools/src/tools/inference/finalize.py`
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
min_accuracy=0.9, max_latency_ms=500.0, promote_path=None) -> FinalizeReport`.

The report is also written as `run_dir/finalize.json`. Export artifacts land
under `run_dir/export/`.

## Behavior

1. Load `config.toml` and locate the processed pack.
2. Score `checkpoints/best.pt` on the test split.
3. Export an FP32 ONNX graph. Export INT8 as well when `int8` is true.
4. Run the golden-scene gate on each exported artifact against the test split.
5. Copy the preferred accepted artifact when `promote_path` is set. INT8 is
   preferred when it passed. FP32 is used otherwise.
6. Write `finalize.json`.

## Errors and faults

`FileNotFoundError` when the run, checkpoint, or pack is missing.
`ImportError` when onnxruntime is missing. `ValueError` from export.

## Messages

None.

## Configuration

Callers pass IoU, accuracy, latency, calibration count, and scene limit.

## Constraints

The live gate needs onnxruntime. INT8 calibration reads the train split of the
same pack the run trained on.

## Related documents

- [`tools.inference.eval`](eval.md)
- [`tools.inference.export`](export.md)
- [`tools.inference.accept`](accept.md)
- [`tools.inference.cli`](cli.md)
