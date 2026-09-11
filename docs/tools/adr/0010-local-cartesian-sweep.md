# ADR-TOOLS-0010: Local cartesian sweep over the run catalog

**Status:** Accepted
**Date:** 2026-08-28
**Topic:** tooling
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-TOOLS-0002, ADR-TOOLS-0006, ADR-TOOLS-0009

## Context

ADR-TOOLS-0006 deferred hyperparameter search. Training now writes unique local
run directories with val metrics and cost fields. Hosted trackers and Optuna
add a second orchestration style and extra dependencies. Lightning remains out
of the tools package.

## Decision

Search is a cartesian product over a TOML space of `TrainConfig` fields. List
values are axes. Scalars are fixed. `max_runs` truncates the product. Each
trial trains, scores the val split, and appends `sweep.jsonl`. Rank sorts local
runs by a val metric and then by FLOPs. Do not add Optuna, Lightning, or a
hosted tracker. Do not read the test split inside the sweep.

## Consequences

- `pact-tools inference sweep --space space.toml` is the search entry.
- `pact-tools inference rank` orders existing runs for a later test eval.
- `known()` lists architecture pairs a space may name.
- A new family is one `nn.Module` builder, one `build()` branch, one registry
  pair, and one test.

## Alternatives considered

- Optuna or Ray Tune — extra stack next to the plain-torch loop.
- Weights & Biases sweep — hosted tracker, rejected by ADR-TOOLS-0006.
- Random search only — less reproducible than a truncated cartesian product.
