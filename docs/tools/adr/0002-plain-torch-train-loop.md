# ADR-TOOLS-0002: Plain torch train loop, no Lightning

**Status:** Accepted
**Date:** 2026-08-26
**Topic:** dependency
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-TOOLS-0001, ADR-REPO-0004

## Context

The dumped industrial-smoke training stack uses PyTorch Lightning, YAML configs,
and pydantic-settings. PACT flight is frozen dataclasses, explicit loops, and a
lean tools default extra set. Lightning would pull a second orchestration style
into `pact-tools`.

## Decision

Train with plain torch: one SGD loop, `BCEWithLogitsLoss`, frozen dataclass or
TOML config, CLI entry only. Do not add Lightning. The `train` extra declares
`torch` and `torchvision`. Torch imports stay lazy inside train/export so
`import tools.model.accept` does not require the extra.

## Consequences

- CI default (`uv sync --extra dev`) stays torch-free.
- Training boxes install `pact-tools[train]`.
- Paper-parity Lightning scripts stay out of this repository.

## Alternatives considered

- Lightning 2.x as in the dump — overkill for one classifier and one U-Net.
- JAX — extra stack with no dump reuse.
