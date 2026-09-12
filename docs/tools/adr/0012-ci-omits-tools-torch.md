# ADR-TOOLS-0012: Lean CI shards omit pact-tools

**Status:** Accepted
**Date:** 2026-09-12
**Topic:** dependency
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-TOOLS-0008, ADR-TOOLS-0011

## Context

ADR-TOOLS-0008 requires `torch` and `torchvision` on `pact-tools` so inference
APIs stay `torch.Tensor` native. Linux CI resolves those packages from PyPI.
The Linux wheel bundles a CUDA 13.0 runtime even though GitHub runners have no
GPU and the tests train tiny CPU models.

Flight, sim, GSE, and analysis tests never import `tools.inference`. Pulling
`pact-tools` on those shards only pays for the torch download. An optional
torch extra on `pact-tools` would break the default CLI: `pact-tools` loads
`tools.inference.cli` at import time, and that module imports torch. A
`cpu`/`cuda` extra pair would split the install matrix that ADR-TOOLS-0011
kept as one resolution.

## Decision

`pact-tools` still requires `torch` and `torchvision`. Default
`uv sync --extra dev` and a training-box `pact-tools` install include them.

CI jobs that do not import tools sync workspace extra `dev-ci-flight`. That
extra omits `pact-tools`, so those shards never download torch. Jobs that
type-check or test tools (`static`, `test-tools`, `test-slow`) still sync
`dev` and install torch.

Tests run on CPU. The Linux wheel may still bundle CUDA. Do not add a torch
optional extra. Do not add a `cpu`/`cuda` extra pair.

## Consequences

- Flight, sim, GSE, and analysis CI shards stay torch-free.
- Tools-facing CI still downloads the PyPI Linux torch wheel.
- The default `pact-tools` CLI keeps working without an extra flag.
- Training boxes keep the GPU path from ADR-TOOLS-0011.

## Alternatives considered

- Optional `inference` extra on `pact-tools` — breaks `pact-tools --help` and
  analysis commands; fights the torch-native API in ADR-TOOLS-0008.
- Linux CPU index or a `cpu` extra — rejected by ADR-TOOLS-0011.
- Lazy-load the inference Typer app — still cannot import `tools.inference`
  without torch.
