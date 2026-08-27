# ADR-TOOLS-0007: INT8 is post-training QDQ PTQ

**Status:** Accepted
**Date:** 2026-08-27
**Topic:** tooling
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-TOOLS-0003

## Context

An INT8 artifact can reduce onboard latency. Operator-style quantization
changes tensor types at the graph boundary. Flight `verify_io_contract` expects
the same float32 NCHW input and output as the FP32 pair. Quantization-aware
training would add a second train path next to the plain SGD loop.

## Decision

INT8 export is post-training static quantization through
`onnxruntime.quantization.quantize_static` with QDQ nodes. Graph input and
output stay float32. Calibration uses a bounded train-split subset from a
processed pack, or synthetic `[0, 1]` NCHW tensors when no pack is given.
The INT8 file is a sibling `*.int8.onnx` with manifest field
`quantization: "int8"`. This path lives behind the `export` extra. Wiring
`InferenceConfig.use_int8` to load that file is a later flight change.

## Consequences

- Accept and I/O-contract checks stay unchanged for INT8 siblings.
- Flight `use_int8` does not select the INT8 file in this change.
- Quantization-aware training, extra architecture families, and dynamic
  quantization stay out until a later decision.

## Alternatives considered

- QOperator quantization — changes graph I/O types and breaks the flight
  contract.
- Quantization-aware training — second train loop and extra hyperparameters.
- Dynamic quantization — no calibration, weaker accuracy control.
