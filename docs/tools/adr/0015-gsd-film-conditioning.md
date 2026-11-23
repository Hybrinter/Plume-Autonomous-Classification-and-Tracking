# ADR-TOOLS-0015: GSD FiLM conditioning in tools training

**Status:** Superseded
**Date:** 2026-11-14
**Topic:** feature-add
**Supersedes:** none
**Superseded-by:** ADR-TOOLS-0017
**Related:** ADR-TOOLS-0002, ADR-TOOLS-0009, ADR-TOOLS-0014

## Context

Finished datasets now carry per-row pixel GSD and the Zenodo source spans
a 10 m native grid plus five flight-elevation bins. A model trained across
those bins sees the same structure at several scales. The `pactnet` and
`dilatenet` flight families need a conditioning input that keeps a single
graph valid across the full GSD range, while the other registered
architectures stay untouched for comparison sweeps.

## Decision

The conditioned families accept a second `(N, 2)` input: the log-ratio
encoding produced by `to_model_gsd` against the shared reference GSD.
`GsdFilm` blocks in `tools.ml_models.arch.film` translate it into
per-channel scale and shift applied after the stem and before the logit
head. The modulation is identity at initialisation, and a missing GSD
argument resolves to the reference encoding, so the legacy single-input
call keeps working until the export cutover.

`arch.registry.build` returns the conditioned graph for `pactnet` and
`dilatenet` and wraps every other family in `IgnoreGsd`, which discards
the GSD argument. Checkpoints record `film-log-gsd-v1` or `ignored` as the
conditioning marker. `tools.ml_models.train` runs the loop over finished
datasets: `config` holds the frozen `TrainConfig`, `provenance` records
training geometry and rejects group leakage across dataset roots,
`evaluate` scores whole splits with a dataset-weighted macro average, and
`loop` writes `checkpoints/best.pt`, `checkpoints/last.pt`, `config.toml`, `summary.json`, and
`history.jsonl` under a unique run directory per ADR-TOOLS-0009. Dataset
weights steer batch sampling only; validation alone selects the best
checkpoint.

## Consequences

- One graph covers every GSD bin; flight inference supplies the same
  encoding at run time.
- Unconditioned baselines remain trainable through the same two-argument
  call.
- Old single-input callers (`tools.inference` training and export) still
  work; a strict two-input graph contract arrives with the export
  cutover.
- Run provenance records actual GSD coverage, shapes, bins, and dataset
  hashes, so a checkpoint states what it trained on.

## Alternatives considered

- Concatenate the encoding as extra input channels. Constant channels
  spend convolution width on an unvarying plane and cannot modulate
  deeper features.
- Train one model per GSD bin. That multiplies artifacts and prevents a
  bin from borrowing data at neighbouring scales.
- Bake sigmoid and fixed thresholds into the graph. Logits stay raw so
  flight keeps threshold authority, unchanged by conditioning.
