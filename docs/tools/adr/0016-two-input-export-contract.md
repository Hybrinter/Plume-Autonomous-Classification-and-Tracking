# ADR-TOOLS-0016: Two-input ONNX export contract and pair gates

**Status:** Accepted
**Date:** 2026-11-18
**Topic:** feature-add
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-TOOLS-0013, ADR-TOOLS-0015

## Context

Training produces conditioned `pactnet`/`dilatenet` checkpoints whose graph
signature is `(image, gsd)`. Flight inference needs a frozen, verifiable
artifact form: the shape contract must live once in flight so tools and the
onboard loader check identical rules, exported artifacts need an auditable
sidecar, and a classifier/segmentor pair must not reach deployment without
acceptance evidence on the finished test split.

## Decision

- `flight.payload.inference.contract` holds the single
  `verify_conditioned_shapes` implementation: dynamic batch, `(None, C, H,
  W)` image, `(None, 2)` GSD, `(None, 1)` or `(None, 1, H, W)` logits.
  `tools.ml_models.export.contract` re-exports it and adds the science-side
  `required_gsd_coverage`/`coverage_ok` formulas over the configured 8x8
  flight tile grid.
- `ModelManifest` is a frozen, extra-forbidden schema-2 JSON sidecar
  (`artifact.with_suffix(".json")`) recording artifact SHA-256, dataset hash,
  declared shapes and float32 dtypes, bands, conditioning/encoding markers,
  GSD reference, actual training GSD coverage, and the fixed tile/grid/frame
  geometry.
- `export` traces `model(image, gsd)` at `(1, 3, 193, 258)` with opset 17,
  `dynamo=False`, dynamic batch, and optional dynamic spatial dims; it writes
  through private sibling temporaries, never overwrites, and never promotes
  to deployment paths. Coverage short of the flight requirement fails unless
  `allow_partial_gsd`, which records `partial_gsd` either way.
- `open_session` verifies the artifact hash before onnxruntime loads it,
  requires exactly `{image, gsd}` float inputs and one float output with a
  shared dynamic batch symbol, and compares normalized declared shapes to the
  manifest. There is no one-input fallback.
- `accept_artifact` gates the session on every test row of every supplied
  finished dataset: classifier `accuracy`, segmentor `mean_iou`, per-source
  thresholds, plus worst batch-one CPU latency. A sibling
  `.acceptance.json` ties the report to artifact SHA-256, dataset hashes, and
  thresholds.
- `write_pair_manifest` requires passing, hash-matched acceptance evidence
  for both artifacts plus matching contract fields before writing a combined
  pair manifest; `flight_promotable` applies the same checks to raw training
  run metadata.

## Consequences

- Graph-contract formulas exist exactly once, in flight; tools cannot drift
  from the onboard verifier.
- Acceptance evidence is mandatory and hash-bound, so stale reports cannot
  back a mutated artifact.
- The legacy one-input `tools.inference` export path and flight loader are
  unchanged; their cutover is #104/#105 scope.
- onnx and onnxruntime stay optional lazy dependencies; the package imports
  and most gates run without them.
