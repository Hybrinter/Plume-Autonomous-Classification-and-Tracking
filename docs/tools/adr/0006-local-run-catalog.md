# ADR-TOOLS-0006: Local filesystem run catalog

**Status:** Accepted
**Date:** 2026-08-27
**Topic:** tooling
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-TOOLS-0002, ADR-TOOLS-0005

## Context

Training needs comparable runs: config, epoch metrics, and checkpoints. Hosted
trackers (W&B, MLflow) and Lightning loggers add a second orchestration style
and extra dependencies. The tools package already writes static SIL report
bundles to disk.

## Decision

Each training job writes a local run directory under `artifacts/runs/<run_id>/`
with `config.toml`, `history.csv`, `checkpoints/last.pt`, `checkpoints/best.pt`,
and `summary.json`. Do not add Weights & Biases, MLflow, or Lightning. Run id
defaults to `{kind}-{arch}-{seed}` and overwrites on repeat.

## Consequences

- Eval, report, and compare commands read these files.
- CI stays free of tracking SaaS credentials.
- Hyperparameter search, extra architecture families, and hosted trackers stay
  out of this package until a later decision.

## Alternatives considered

- Weights & Biases or MLflow — extra account, network, and a second source of
  truth next to git.
- PyTorch Lightning loggers — conflicts with the plain-torch train loop.
