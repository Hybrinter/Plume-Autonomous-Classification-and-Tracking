# ADR-TOOLS-0017: Single-source unit-tile workflow

**Status:** Accepted
**Date:** 2026-10-02
**Topic:** restructure
**Supersedes:** ADR-TOOLS-0014, ADR-TOOLS-0015
**Superseded-by:** none
**Related:** ADR-TOOLS-0002, ADR-TOOLS-0004, ADR-TOOLS-0013, ADR-TOOLS-0016

## Context

The earlier finished-dataset design let training, acceptance, and
calibration mix several finished datasets through `check_compatible` and
explicit dataset weights. That plural path added a second agreement
surface: cross-root manifest comparison, weighted macro metric
aggregation, and dataset-choice sampling all duplicated checks the one
build already records.

In parallel, raw tiles still carried source-implied geometry: adapters
reported `extent_m`/`bit_depth`, and the ground pipeline normalized,
clipped, and quantized pixels. That made the same image pass through two
different contracts — one at the source, one on disk — and let flight
geometry leak into Zenodo bin selection.

## Decision

- Pixels are float32 unit values end to end. Sources return float32
  `(C, H, W)` images in `[0, 1]`; the build permutes them for
  augmentation and writes them unchanged. Normalization, clipping, and
  image quantization are removed from the generic ground build and
  loader; Zenodo retains its source-specific counts-to-reflectance
  scaling and camera spectral-response table. Normalization metadata is
  the fixed `norm="unit"` marker.
- Geometry is indexed, not derived. Each `RawTileRef` carries its actual
  `height`/`width`; shard directories and rows record the real tile
  shape. Finished datasets and flight tile imports are schema 2 and
  record `image_dtype="float32"`.
- Zenodo GSD bins are a fixed source-owned table — a native 10 m grid
  and approximate targets at 15, 20, 25, 30, and 35 m over the 1200 m
  extent — with actual GSD stored as extent over integer output
  dimensions. The spatial bins — not the rest of the adapter — are
  independent of flight orbit, optics, and elevation geometry; the
  spectral response matching is explicitly retained.
- Training, evaluation, acceptance, and calibration each consume exactly
  one finished dataset root. `TrainConfig` carries one `dataset` string;
  the legacy `datasets`/`dataset_weights` keys are rejected. The CLI
  accepts a repeatable `--dataset` only to reject a second root
  explicitly.
- `check_compatible`, dataset weights, and weighted macro aggregation
  are deleted. The one dataset's agreement with a model — bands, unit
  pixel domain, GSD reference, and fixed artifact input shape — is
  checked inline at acceptance and calibration; incompatible datasets
  fail rather than being resized.
- Training provenance records one `provenance["dataset"]` object and
  uses the manifest's `dataset_hash` directly. Measured GSD bounds come
  only from rows not marked `gsd_nominal`; nominal rows still count
  toward `train_samples`. The misleading flight trace-size checkpoint
  fields are removed; actual `spatial_shapes` are recorded instead.
- Ground full-frame evaluation is removed: `analysis/full_frame.py`,
  the `frame-eval` command, `dataset/geometry.py`, and their tests and
  documentation. The retained `GsdFilm` conditioning interface, the
  two-input export contract, acceptance thresholds, pair promotion, and
  flight slice/stitch behavior are unchanged.

## Consequences

One run depends on one auditable dataset; its hash alone identifies the
training content. Group-disjoint splits, per-bin exhaustive evaluation,
round-robin calibration over same-root size shards, and seeded
batch order are preserved. Datasets that disagree with an artifact on
bands, domain, reference, or fixed shape fail clearly at the consuming
boundary. Flight-side evaluation of stitched frames, when needed, is a
flight concern and no longer duplicated on the ground.
