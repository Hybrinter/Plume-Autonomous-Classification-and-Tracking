# ADR-TOOLS-0014: Finished datasets per source

**Status:** Accepted
**Date:** 2026-10-01
**Topic:** restructure
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-TOOLS-0004, ADR-TOOLS-0013

## Context

`tools.ml_models` previously carried a processed-pack layer (`data`) that
concatenated image tensors, masks, labels, and split indices into one pack
per training run. Model work now consumes several origins of tiles — a
labeled flight tile directory and a synthetic generator — with more
expected. Each origin has its own row count, GSD coverage, and provenance.
Merging origins at pack time hides those differences and forces a rebuild
of every pack when one origin changes.

## Decision

Each raw tile source builds its own finished dataset under
`tools.ml_models.dataset`. A source implements the `RawSource` protocol:
an index of `RawTileRef` rows plus a forward-only `iter_tiles` stream.
`build_dataset` applies one `BuildSpec` (group split, train-only dihedral
augmentation, tasks, band list, GSD reference) to any source and writes
`.npy` shards under `<task>/<split>/<H>x<W>` plus a `dataset.json`
manifest with a content hash. Mixing happens at load time: `make_loader`
draws batches across finished datasets after `check_compatible` confirms
shared bands, unit norm, and GSD reference. The old `meta`, `norm`, and
`pack` modules are removed; `split` moves under `dataset`. The build
command lives at `python -m tools.ml_models dataset build` and is mounted
on the root tools CLI as `ml-models`.

## Consequences

- One source change rebuilds one dataset; other dataset roots stay valid.
- Provenance (`source`, `source_ref`, bins, recipes) is recorded per
  dataset in `dataset.json`, and `dataset_hash` verifies the shard files.
- New origins need only a `RawSource` implementation; no new build code.
- The `tools.inference` pack readers and `tools.original_dataset_analysis`
  stay importable until a later change removes them.
- Training corpora and finished datasets stay outside git, as
  ADR-TOOLS-0004 requires.

## Alternatives considered

- Keep a single merged pack per training run. Every origin change would
  invalidate the whole pack and per-source provenance would be lost.
- Mix sources inside the build. A build would need a multi-source index
  and the split would no longer be reproducible per origin.
- Keep `data` beside `dataset`. Two pack layouts would compete until the
  inference stack migrates.
