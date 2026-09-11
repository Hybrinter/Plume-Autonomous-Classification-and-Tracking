# ADR-TOOLS-0008: Torch is a required tools dependency

**Status:** Accepted
**Date:** 2026-08-28
**Topic:** dependency
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-TOOLS-0002

## Context

Inference data, metrics, and architecture work lived behind an optional train
extra so default CI stayed torch-free. In-memory packs were numpy arrays.
Designing new `nn.Module` architectures then required a numpy-to-torch
boundary at every Dataset, metric, and train step.

## Decision

`pact-tools` requires `torch` and `torchvision`. Inference tool APIs use
`torch.Tensor`, `torch.utils.data.Dataset`, and `DataLoader`. On-disk processed
packs stay `.npy`. NumPy remains at disk I/O, matplotlib, and onnxruntime
edges. CI `uv sync --extra dev` installs torch. Flight does not depend on
tools and stays torch-free. Do not add Lightning.

## Consequences

- Default tools install includes the CPU torch wheel.
- The `train` extra is gone. The `export` extra still adds onnx and
  onnxruntime.
- New architectures are one `nn.Module` plus a registry row.

## Alternatives considered

- Keep torch optional and lazy-import — leaves a numpy Dataset surface.
- Add Lightning — second orchestration style next to the plain SGD loop.
