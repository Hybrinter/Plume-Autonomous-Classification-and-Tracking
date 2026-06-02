# ADR 0004: ONNX frozen-artifact detector behind a swappable backend

**Status:** Accepted (2026-05-30)

## Context

The onboard model performs plume detection/segmentation. The legacy approach ran a torch
`nn.Module` directly in flight, pulling torch (and its weight) into the flight image, and mixed
inference with the training code. The model must never be trained or mutated in flight (weights are
frozen after construction), and CI must run without GPU/ML SDKs.

## Decision

Define a small **`DetectorBackend` Protocol** (`detect(frame) -> Result[InferenceResultMsg, ...]`)
with two implementations: **`OnnxDetector`** -- a frozen, versioned `.onnx` artifact run via
`onnxruntime` (lazy-imported) -- for flight, and **`ScriptedDetector`** -- a fixed probability mask
-- for SIL and tests. Both share `extract_blobs` so detection geometry is identical. Training and
export-to-ONNX live entirely in `tools/` (free-form torch), out of the flight image.

## Consequences

- The flight image carries no torch; `onnxruntime` is an optional, lazily-imported dependency, so
  CI and `import-linter` stay SDK-free and the absent-runtime path is exercised.
- The model is a configuration-controlled artifact whose version/hash is recorded in telemetry.
- `ScriptedDetector` gives deterministic detections, which is what makes the SIL closed-loop test
  reproducible.
- A frozen torch module remains an acceptable alternative backend; this is no longer a load-bearing
  decision now that Rust is off the table.
