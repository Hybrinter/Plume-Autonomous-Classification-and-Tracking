# ADR-TOOLS-0011: Windows tools installs CUDA torch wheels

**Status:** Accepted
**Date:** 2026-08-28
**Topic:** dependency
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-TOOLS-0002, ADR-TOOLS-0008

## Context

ADR-TOOLS-0008 made `torch` and `torchvision` required `pact-tools`
dependencies resolved from PyPI. The PyPI Linux wheel for torch 2.13 carries a
bundled CUDA 13.0 runtime, so a Linux training box reaches the GPU from the
default install. The PyPI Windows wheel is CPU-only. On a Windows training box
with an NVIDIA GPU, `torch.cuda.is_available()` returns false and the train loop
in `tools.inference.train` falls back to CPU. Training a segmentor on CPU is
slower than the GPU path by a large factor.

PyTorch publishes accelerator builds to per-CUDA indexes rather than PyPI. The
CUDA 13.0 index carries `win_amd64` wheels for torch 2.13.0 and torchvision
0.28.0, which matches the CUDA 13.0 runtime the Linux PyPI wheel already pins.
The wheels bundle the CUDA runtime, so a Windows box needs an NVIDIA driver but
no separate CUDA toolkit install.

## Decision

`packages/tools/pyproject.toml` declares the CUDA 13.0 PyTorch index
`https://download.pytorch.org/whl/cu130` with `explicit = true`, and routes
`torch` and `torchvision` to it under the marker `sys_platform == 'win32'`.
Linux and macOS keep resolving both packages from PyPI. Version floors stay
`torch>=2.2` and `torchvision>=0.17`.

Windows installs require an NVIDIA driver that supports CUDA 13.0 or later. Do
not add a second CUDA index, a `cpu`/`cuda` extra pair, or a conflicting-extra
matrix. Do not route Linux to a PyTorch index while the PyPI wheel ships CUDA.

## Consequences

- `uv.lock` holds two resolutions per package: `2.13.0` from PyPI for
  `sys_platform != 'win32'` and `2.13.0+cu130` from the CUDA index for
  `sys_platform == 'win32'`.
- A Windows training box gets the GPU from `uv sync` with no extra flag.
- CI stays on `ubuntu-latest` and keeps resolving torch from PyPI, so the gate
  set and its download size do not change.
- macOS keeps the PyPI wheel, which has no CUDA build.
- A CUDA version bump is one index URL edit plus a relock.

## Alternatives considered

- `cpu` and `cu130` extras with `tool.uv` conflicts — a second install mode and
  a per-box flag for a single supported training platform.
- Route every platform to the CUDA index — replaces working Linux PyPI wheels
  and adds a non-PyPI index to CI.
- `uv pip install --torch-backend=auto` outside the lock — leaves the resolved
  accelerator out of `uv.lock` and off the record.
- Document a manual post-sync reinstall — drifts from the lock on every sync.
