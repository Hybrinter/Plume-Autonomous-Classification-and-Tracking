# Design briefs

**Kind:** architecture briefs (not as-built)

## Purpose

This directory holds architecture briefs for work that is not yet the running system.
Descriptive pages under `docs/flight`, `docs/sim`, `docs/gse`, and `docs/tools` stay
as-is. Point an implementation agent at a brief here; do not treat today's modules
as the specification for work that still lives only in a brief.

## Contents

| Document | Topic |
| --- | --- |
| [`design/single-axis-elevation-controller.md`](design/single-axis-elevation-controller.md) | Single-axis elevation gimbal controller |
| [`design/scene-twin.md`](design/scene-twin.md) | Mix-and-match world twin (named physics models) |

## Constraints

- Briefs may use equations and future-tense design language.
- Briefs must not cite architecture-decision identifiers. The ADR scanner walks the
  whole `docs/` tree.
- After an implementation lands, update the STE-mirrored pages to match the code.
  Do not leave the brief as the only description of as-built behavior.
