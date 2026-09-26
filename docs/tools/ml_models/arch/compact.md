# tools.ml_models.arch.compact

**Source:** `packages/tools/src/tools/ml_models/arch/compact.py`
**Kind:** module

## Purpose

This module defines a compact binary classifier family named `pactnet`. The stack
uses depthwise-separable convolutions and early downsampling. The default head
is adaptive average pooling and a linear layer. The `max` token uses a 1x1
convolution and returns the maximum logit over the strided cells.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `COMPACT_PREFIX` | constant | Family prefix string `pactnet` |
| `DEFAULT_COMPACT_WIDTH` | constant | Default stem width (16) |
| `DEFAULT_COMPACT_DEPTH` | constant | Default stage count (4) |
| `CompactSpec` | class | Parsed width, depth, convolution style, and spatial head |
| `compact_stage_widths` | function | Per-stage channel counts for a width and depth |
| `PactNet` | class | Compact separable convolution stack |
| `parse_compact` | function | Parse a `pactnet` registry name into a `CompactSpec` |
| `build_compact_classifier` | function | Construct a `PactNet` from a spec |

## Inputs and outputs

`compact_stage_widths(base_width, depth) -> tuple[int, ...]`. Returns `depth`
widths. Each stage doubles the channel count up to a ceiling of 256.

`parse_compact(name) -> CompactSpec`. Raises `ValueError` on an unknown family
or modifier token.

`PactNet.spatial(x)` maps `(N, C, H, W)` to `(N, 1, h, w)` logits. The `max`
token selects this head.

`PactNet.forward(x)` returns shape `(N, 1)`. With `max`, the value is
`spatial(x).amax(dim=(2, 3))`. Without `max`, the value is the linear head
after adaptive average pooling. No sigmoid is applied.

`build_compact_classifier(spec, in_channels=3) -> PactNet`.

## Behavior

1. Registry names use the `pactnet` prefix with underscore-separated modifiers.
2. `w<N>` sets the stem width. Default is 16.
3. `d<N>` sets the strided stage count including the stem. Default is 4.
4. `full` selects dense 3x3 convolutions. The default uses depthwise-separable
   convolutions.
5. The stem convolution is always dense. Later stages honour the `separable`
   flag.
6. Each stage after the stem applies a strided block and a 1x1-stride block.
7. Without `max`, the head is adaptive average pooling, flatten, dropout, and
   a linear layer. `head.weight` has rank 2.
8. With `max`, dropout applies on the feature map. A 1x1 convolution emits one
   logit per cell. `forward` returns the maximum over those cells.
   `head.weight` has shape `(1, C, 1, 1)`.
9. Modifiers combine in any order. Examples: `pactnet_w32_d5_full` and
   `pactnet_max_d5_full`.

## Errors and faults

`ValueError` on an unknown family, modifier token, or a width or depth below one.

## Messages

None.

## Configuration

`in_channels` defaults to 3 (BLUE, GREEN, RED). Head dropout is fixed at 0.2.
Without `max`, dropout follows the pooled vector. With `max`, dropout applies
on the feature map. The maximum stage width is 256.

## Constraints

This module imports torch at import time. The graph does not apply sigmoid.
Registry names use the compact classifier grammar in
[`tools.ml_models.arch.registry`](registry.md).

## Related documents

- [`tools.ml_models.arch`](../arch.md)
- [`tools.ml_models.arch.grammar`](grammar.md)
- [`tools.ml_models.arch.blocks`](blocks.md)
- [`tools.ml_models.arch.classifier`](classifier.md)
- [`tools.ml_models.arch.registry`](registry.md)
- [`tools.inference.train`](../../inference/train.md)
