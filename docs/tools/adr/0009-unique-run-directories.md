# ADR-TOOLS-0009: Unique run directories refuse overwrite

**Status:** Accepted
**Date:** 2026-08-28
**Topic:** tooling
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-TOOLS-0006

## Context

ADR-TOOLS-0006 stores each training job under `artifacts/runs/<run_id>/`. The
default run id `{kind}-{arch}-{seed}` overwrites that directory on a repeat.
Hyperparameter search needs comparable history. A silent overwrite deletes the
prior config, metrics, and checkpoints.

## Decision

Default `run_id` is `{kind}-{arch}-{seed}-{digest8}`. `digest8` is the first
eight hex characters of SHA-256 over the JSON of `TrainConfig` fields except
`run_dir`, `run_id`, `checkpoint_path`, and `overwrite`. A caller-supplied
`run_id` is used unchanged. `train()` raises `FileExistsError` when the run
directory already contains `summary.json`, unless `overwrite` is true. Do not
add a hosted tracker.

## Consequences

- Repeat jobs with distinct hyperparameters land in distinct directories.
- An explicit `run_id` stays stable for an agent-supplied name.
- `--overwrite` and `TrainConfig.overwrite` replace a named run.
- Hyperparameter search remains a later decision.

## Alternatives considered

- Timestamp suffix — less reproducible across machines.
- UUID suffix — no link from config to directory name.
- Always overwrite — loses prior trials during a search.
