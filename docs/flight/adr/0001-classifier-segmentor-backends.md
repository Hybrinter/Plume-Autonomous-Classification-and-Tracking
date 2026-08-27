# ADR-FLIGHT-0001: ClassifierBackend, SegmentorBackend, and blob DetectorBackend

**Status:** Accepted
**Date:** 2026-08-26
**Topic:** interface
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-REPO-0004, ADR-TOOLS-0003

## Context

`OnnxDetector` mixed a segmentation ONNX session with blob extraction under the
name "detector". The science pipeline needs a binary presence classifier that
can skip the segmentor, and a segmentor that emits a mask. The payload app still
calls one `detect()` per frame.

## Decision

- `ClassifierBackend.classify` returns a logit and a boolean gate.
- `SegmentorBackend.segment` returns an `(H, W)` probability mask.
- `DetectorBackend.detect` composes classifier, then maybe segmentor, then
  `extract_blobs`.
- `ScriptedDetector` defaults to always-positive and `inference_ms = 0.0`.
- `OnnxDetector` loads two frozen `.onnx` files.

## Consequences

- Negative classification yields empty blobs and a zero mask. Control stays in
  search or scan.
- SIL pointing tests keep a scripted always-positive classifier.
- Model deploy is pair-only: one uplink blob must carry both classifier and
  segmentor contracts. Either I/O failure rolls back the pair.

## Alternatives considered

- Two `detect()` calls in `PayloadApp` — splits the co-located inference path.
- Classifier-only filter in preprocessing — mixes quality flags with ML.
