# ADR-TOOLS-0018: Evidence-first phased ML analysis

**Status:** Accepted
**Date:** 2026-10-05
**Topic:** restructure
**Supersedes:** ADR-TOOLS-0006, ADR-TOOLS-0009
**Superseded-by:** none
**Related:** ADR-TOOLS-0002, ADR-TOOLS-0013, ADR-TOOLS-0016, ADR-TOOLS-0017

## Context

The standard training loop and legacy report writer use different history and summary contracts. Existing scalar scores omit validity and support information, and report previews do not establish whole-population failure behavior. Dataset variants and augmented rows also require explicit observation and group provenance. Scientific measurements must remain stable when figures are restyled.

## Decision

Separate dataset generation, dataset analysis, training, post-training analysis, and export. One shared evaluator and metric implementation produces versioned evidence. Plotting and visuals consume captured evidence and never rerun inference implicitly.

Each dataset-analysis execution and each model/training-analysis execution writes one tagged summary with supporting tables, curves, and visual artifacts. Training writes execution metadata, incremental history, and export-compatible checkpoints. Output directories are unique and existing outputs are refused, including publication races. Keep reports outside hashed dataset roots and retain local filesystem ownership without a hosted tracker.

Keep model training and operating-point selection separate from final-test evaluation. Training does not score test; final analysis identifies the frozen checkpoint and settings. Report unavailable measurements with reasons instead of fabricated zeros. Default capture is compact; full dense predictions are optional.

Replace the obsolete implementation through a sequential feature stack. During the scaffold phase, unimplemented training, evaluation, analysis, and quality acceptance return explicit unavailable errors. There is no legacy scoring fallback and no acceptance success from placeholders.

## Consequences

Measurements have explicit populations, definitions, thresholds, support, and provenance. Figures can be regenerated without changing their scientific inputs. The transition temporarily removes standard training and quality acceptance until their replacement phases land. Dataset generation and export serialization remain available. Individual-model evidence does not establish paired-pipeline or target-hardware flight qualification.

## Alternatives considered

- Retain duplicate legacy scoring behind the new CLI: preserves ambiguous score definitions and permits divergent results.
- Recompute metrics during plotting: changes evidence when presentation changes and repeats inference.
- Add a hosted tracker or training framework: adds dependencies and another source of truth without resolving the evidence contract.
