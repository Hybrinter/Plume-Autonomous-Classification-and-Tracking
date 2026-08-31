# tools.inference.arch.compact

**Source:** `packages/tools/src/tools/inference/arch/compact.py`
**Kind:** module

## Purpose

This module defines a compact binary classifier family named `pactnet`. The stack
uses depthwise-separable convolutions, aggressive early downsampling, and global
average pooling into a single linear head. Output is one logit per tile.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `COMPACT_PREFIX` | constant | Family prefix string `pactnet` |
| `DEFAULT_COMPACT_WIDTH` | constant | Default stem width (16) |
| `DEFAULT_COMPACT_DEPTH` | constant | Default stage count (4) |
| `CompactSpec` | class | Parsed width, depth, and convolution style |
| `compact_stage_widths` | function | Per-stage channel counts for a width and depth |
| `PactNet` | class | Compact separable convolution stack |
| `parse_compact` | function | Parse a `pactnet` registry name into a `CompactSpec` |
| `build_compact_classifier` | function | Construct a `PactNet` from a spec |

## Inputs and outputs

`compact_stage_widths(base_width, depth) -> tuple[int, ...]`. Returns `depth`
widths. Each stage doubles the channel count up to a ceiling of 256.

`parse_compact(name) -> CompactSpec`. Raises `ValueError` on an unknown family
or modifier token.

`PactNet.forward(x)` maps `(N, C, H, W)` to `(N, 1)` logits. No sigmoid is
applied.

`build_compact_classifier(spec, in_channels=4) -> PactNet`.

## Behavior

1. Registry names use the `pactnet` prefix with underscore-separated modifiers.
2. `w<N>` sets the stem width. Default is 16.
3. `d<N>` sets the strided stage count including the stem. Default is 4.
4. `full` selects dense 3x3 convolutions. The default uses depthwise-separable
   convolutions.
5. The stem convolution is always dense. Later stages honour the `separable`
   flag.
6. Each stage after the stem applies a strided block and a 1x1-stride block.
7. Adaptive average pooling feeds a dropout layer and a single linear head.
8. Modifiers combine in any order. Example: `pactnet_w32_d5_full`.

## Errors and faults

`ValueError` on an unknown family, modifier token, or a width or depth below one.

## Messages

None.

## Configuration

`in_channels` defaults to 4. Head dropout is fixed at 0.2. The maximum stage
width is 256.

## Constraints

This module imports torch at import time. The graph does not apply sigmoid.
Registry names use the compact classifier grammar in
[`tools.inference.arch.registry`](registry.md).

## Related documents

- [`tools.inference.arch`](../arch.md)
- [`tools.inference.arch.grammar`](grammar.md)
- [`tools.inference.arch.blocks`](blocks.md)
- [`tools.inference.arch.classifier`](classifier.md)
- [`tools.inference.arch.registry`](registry.md)
- [`tools.inference.train`](../train.md)
