# ADR-TOOLS-0003: Two frozen ONNX artifacts with a classifier filter

**Status:** Accepted
**Date:** 2026-08-26
**Topic:** interface
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-FLIGHT-0001, ADR-REPO-0004, ADR-TOOLS-0001

## Context

Flight previously assumed one segmentation ONNX. The science pipeline has a
binary presence classifier (one plume class) and a U-Net-class segmentor. They
are trained and exported as separate graphs. A combined graph would run the
segmentor on empty frames.

## Decision

Export two artifacts: classifier `(1, C, H, W) -> (1, 1)` logit and segmentor
`(1, C, H, W) -> (1, 1, H, W)` logits. Do not bake sigmoid into either graph.
Spatial size comes from `InferenceConfig`, default 256. Domain is
`normalize_dn` on BLUE/GREEN/RED/NIR. The classifier gates the segmentor on
board. Acceptance uses mask IoU for the segmentor and binary accuracy for the
classifier.

## Consequences

- `data/models/` holds `active_classifier.onnx` and `active_segmentor.onnx`.
- SIL scripted detector stays always-positive and does not load ONNX.
- Input height and width remain config fields for later Orin retune.

## Alternatives considered

- One combined ONNX graph — always pays segmentor cost.
- Paper 120 px mean/std domain as the flight contract — mismatches preprocess.
