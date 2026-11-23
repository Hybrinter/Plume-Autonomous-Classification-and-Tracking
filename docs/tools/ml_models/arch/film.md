# tools.ml_models.arch.film

**Source:** `packages/tools/src/tools/ml_models/arch/film.py`
**Kind:** module

## Purpose

This module carries the GSD conditioning shared by the `pactnet` and
`dilatenet` families. A `GsdFilm` block maps the `(N, 2)` encoded GSD to a
per-channel scale and shift applied to a feature map. `IgnoreGsd` adapts
single-input registry graphs to the two-argument call.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `CONDITIONING_ID` | constant | Checkpoint marker `film-log-gsd-v1` |
| `IGNORED_CONDITIONING_ID` | constant | Checkpoint marker `ignored` |
| `GsdModel` | protocol | `model(image, gsd) -> logits` call signature |
| `GsdFilm` | class | FiLM block for one feature-map channel count |
| `IgnoreGsd` | class | Two-argument adapter over a single-input graph |
| `resolve_gsd` | function | Fallback encoding when `gsd` is None |

## Inputs and outputs

`GsdFilm(channels)` builds a sequential `Linear(2, 16) -> ReLU ->
Linear(16, 2 * channels)` head.

`GsdFilm.forward(features, gsd)`: `features` is `(N, C, H, W)`; `gsd` is
`(N, 2)`. The head output splits into `delta_gamma` and `beta` of shape
`(N, C)` each; the result is
`features * (1 + delta_gamma[..., None, None]) + beta[..., None, None]`.

`resolve_gsd(x, gsd)` returns `gsd` or a zeros `(N, 2)` tensor on `x`'s
device and dtype.

## Behavior

1. The final linear layer starts at zero weight and bias, so a fresh
   `GsdFilm` is an identity map for any input GSD.
2. Zeros are the log-ratio encoding of the reference GSD, so a None
   argument behaves as a tile at the reference.
3. `IgnoreGsd.forward(image, gsd=None)` returns `inner(image)`; the GSD
   tensor is discarded.
4. `CONDITIONING_ID` marks checkpoints of the conditioned families;
   wrapped graphs record `ignored`.

## Errors and faults

None raised directly. Shape mismatches surface from `nn.Linear`.

## Messages

None.

## Configuration

The hidden width is fixed at 16.

## Constraints

This module imports torch at import time. It does not import `flight`.

## Related documents

- [`tools.ml_models.arch`](../arch.md)
- [`tools.ml_models.arch.compact`](compact.md)
- [`tools.ml_models.arch.dilated`](dilated.md)
- [`tools.ml_models.arch.registry`](registry.md)
- [`tools.ml_models.dataset.gsd`](../dataset/gsd.md)
