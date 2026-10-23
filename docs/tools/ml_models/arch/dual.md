# tools.ml_models.arch.dual

**Source:** `packages/tools/src/tools/ml_models/arch/dual.py`
**Kind:** module

## Purpose

This module defines dual-branch variants of the `pactnet` classifier and the
`dilatenet` segmentor for a BLUE, GREEN, RED, SWIR input. A VIS branch reads
planes 0-2 and a SWIR branch reads plane 3. The branches are fused at output
stride 4, and the rest of the network is the usual family body.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DUAL_INPUT_CHANNELS` | constant | Required input plane count (4) |
| `DEFAULT_JITTER_PX` | constant | Default training SWIR shift limit (1 VIS sample) |
| `DEFAULT_MODALITY_DROPOUT` | constant | Default per-branch training drop probability (0.1) |
| `shift_planes` | function | Per-sample integer shift with edge replication |
| `DualStem` | class | VIS and SWIR branch stems fused at stride 4 |
| `DualPactNet` | class | Dual-branch compact classifier |
| `DualDilatedSegmentor` | class | Dual-branch dilated segmentor |
| `build_dual_classifier` | function | Construct a `DualPactNet` from a `CompactSpec` |
| `build_dual_segmentor` | function | Construct a `DualDilatedSegmentor` from a `DilatedSpec` |

## Inputs and outputs

`DualPactNet.forward(x, gsd=None)` maps `(N, 4, H, W)` to `(N, 1)` logits.
`DualPactNet.spatial(x, gsd=None)` returns the `(N, 1, h, w)` cell logits.
`DualDilatedSegmentor.forward(x, gsd=None)` maps `(N, 4, H, W)` to
`(N, out_channels, H, W)` logits on the VIS grid. `gsd` is the `(N, 2)` VIS
GSD encoding; None selects the reference GSD.

`shift_planes(planes, offsets, max_px)` shifts each sample of `(N, C, H, W)` by
its `(dy, dx)` row of `offsets` and replicates the edge pixels.

## Behavior

1. Each branch is a dense stride-2 stem, a `GsdFilm`, and one stride-2 block.
   The SWIR stem is half the VIS stem width, at least 1.
2. Both branches receive the VIS GSD encoding.
3. A 1x1 convolution, batch-norm, and ReLU mix the concatenated branch
   features to the fused width. The classifier fuses to the second `pactnet`
   stage width. The segmentor fuses to the `dilatenet` body width.
4. `DualPactNet` then applies one stride-1 block, the remaining `pactnet`
   stages, a head `GsdFilm`, feature-map dropout 0.2, a 1x1 head, and a max
   over cells.
5. `DualDilatedSegmentor` then applies one strided block at output stride 8,
   the dilated blocks, a head `GsdFilm`, a 1x1 head, and a bilinear resize to
   the input size.
6. In training mode, the SWIR plane of each sample shifts by a random integer
   offset in `[-jitter_px, jitter_px]` per axis before the SWIR stem.
7. In training mode, a sample loses its VIS branch features with probability
   `modality_dropout` and its SWIR branch features with the same probability.
   A sample never loses both.
8. In eval mode, the shift and the dropout do not apply.

## Errors and faults

`ValueError` when the input band count is not 4, a `pactnet_dual` depth is
below 2, a `dilatenet_dual` output stride is not 4 or 8, a width is below 1,
`jitter_px` is negative, or `modality_dropout` is outside `[0, 0.5]`.

## Messages

None.

## Configuration

Registry names are `pactnet_dual` and `dilatenet_dual` with the family's other
modifiers, for example `pactnet_dual_w32` or `dilatenet_dual_s8`. The names do
not set `jitter_px`; the registry uses the default. The `nodrop` token sets
`modality_dropout` to 0, for example `pactnet_dual_nodrop`. Without `nodrop`,
the registry uses the default `modality_dropout`.

## Constraints

The input plane order must be BLUE, GREEN, RED, SWIR, with SWIR on the VIS
grid, as the paired Zenodo build writes it. ONNX export of the dual variants
is not verified. This module imports torch at import time.

## Related documents

- [`tools.ml_models.arch`](../arch.md)
- [`tools.ml_models.arch.compact`](compact.md)
- [`tools.ml_models.arch.dilated`](dilated.md)
- [`tools.ml_models.arch.film`](film.md)
- [`tools.ml_models.arch.registry`](registry.md)
- [`tools.ml_models.dataset.sources.zenodo.paired`](../dataset/sources/zenodo/paired.md)
