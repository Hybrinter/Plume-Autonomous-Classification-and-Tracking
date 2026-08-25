# ADR-REPO-0011: Ship flight as an isolated package

**Status:** Accepted
**Date:** 2026-08-25
**Topic:** dependency
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-REPO-0002, ADR-REPO-0004

## Context

The repository documents a single install path: `uv sync --extra dev` over the whole workspace.
That path pulls every package and every dev dependency. Training and export stacks (PyTorch, JAX,
and similar heavy ML frameworks) must never land on the payload computer. Git cherry-picks and
sparse checkouts filter by file path, not by dependency closure. They do not guarantee a lean
flight image.

ADR-REPO-0002 established a `uv` workspace with lean flight and heavy tools packages.
ADR-REPO-0004 established a frozen ONNX artifact as the boundary between training and flight
inference. This record addresses how those packages are installed and shipped, not which inference
runtime runs onboard.

## Decision

Ship the flight experiment as the **`pact-flight` wheel**. Hatch packs only `src/flight` into
that wheel. The same git repository supports three install profiles:

| Profile | Install | Purpose |
| --- | --- | --- |
| Laptop / CI | `uv sync --extra dev` | Full workspace, dev tools, all gates |
| Training / export | `pact-tools` with extras `train` and `export` | Model training and artifact export |
| Flight experiment | `pact-flight` with extras `inference`, `camera`, `gimbal` | Payload computer or experiment image |

Optional extras are **named by role**, not by vendor. They may start empty and gain packages when
a stack is chosen. Examples: `inference` for the onboard runtime, `train` for the training
framework, `export` for export tooling, `camera` and `gimbal` for pip-installable device SDKs.

The boundary object between training and flight remains a **frozen model artifact** (file, hash,
and I/O contract), not a Python training module. This ADR does not change the inference backend
decision in ADR-REPO-0004.

CI proves a flight-only virtual environment: no `pact-tools`, `pact-sim`, or `pact-gse`, and no
training frameworks. Vendor SDKs that are not on PyPI stay out of extras; they belong in the
image layer or a vendor install step.

**Out of scope for this ADR:** Docker image layout, git promotion branches, and pinning specific
versions of torch, JAX, or onnxruntime.

## Consequences

- The payload computer installs only `pact-flight` (plus chosen extras), not the whole workspace.
- Role extras give a stable install surface before a vendor stack is selected.
- CI can fail when a flight-only venv picks up training or non-flight packages.
- Training boxes install `pact-tools[train]` (and later `[export]`) without touching flight deps.
- Vendor SDKs outside PyPI remain an image-layer concern, not a pip extra.

## Alternatives considered

- **Git cherry-pick or sparse-checkout** — filters files, not dependency closure; easy to miss a
  transitive import or a stray dev dependency.
- **Separate training repository** — hard fork between training and flight code; duplicate review
  and drift risk for shared types and the artifact contract.
- **Bake the whole workspace into the flight image** — simplest install story, but guarantees
  heavy ML and non-flight packages on the payload computer.
