# ADR-TOOLS-0013: One tools.ml_models package for train, export, and accept

**Status:** Accepted
**Date:** 2026-09-26
**Topic:** restructure
**Supersedes:** ADR-TOOLS-0005
**Superseded-by:** none
**Related:** ADR-TOOLS-0001, ADR-TOOLS-0004

## Context

ADR-TOOLS-0005 placed model training, export, and acceptance in
`tools.inference`, and kept `tools.analysis` as the SIL recorder. Band and
ground-sample-distance studies then landed in `tools.original_dataset_analysis`.
Those workflows share processed packs, splits, and normalization. They do not
share the SIL recorder.

## Decision

Model training, export, acceptance, and band studies live in `tools.ml_models`.
`tools.inference` and `tools.original_dataset_analysis` remain until a later
change removes them. `tools.analysis` stays the SIL recorder. Flight does not
import tools. Training bytes stay out of git.

## Consequences

- New training, export, acceptance, and band-study code imports
  `tools.ml_models`.
- `tools.inference` and `tools.original_dataset_analysis` stay importable until
  a later change removes them.
- `tools.analysis` remains the deterministic SIL capture path.
- Flight still does not import `tools`.
- Training corpora and processed packs stay outside git, as ADR-TOOLS-0004
  requires.

## Alternatives considered

- Leave model workflows only in `tools.inference`. The Zenodo study would stay
  in a second package with a second pack codec.
- Remove `tools.inference` in this change. Train and export would disappear
  before `tools.ml_models` provides those modules.
- Put band studies in `tools.analysis`. Dataset preparation would sit in the
  SIL recorder.
