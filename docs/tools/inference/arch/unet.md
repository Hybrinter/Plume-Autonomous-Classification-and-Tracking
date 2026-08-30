# tools.inference.arch.unet

**Source:** `packages/tools/src/tools/inference/arch/unet.py`
**Kind:** module

## Purpose

This module defines a clean-room encoder-decoder segmentor. `base_width`,
`depth`, and `separable` parameterise the family. Defaults reproduce the
original 64-128-256-512 bilinear U-Net. Output is a logit map.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ENCODER_CHANNELS` | constant | Default `(64, 128, 256, 512)` stage widths |
| `stage_widths` | function | Per-stage channel counts for a width and depth |
| `ConvBlock` | class | Two 3x3 conv-BN-ReLU stages |
| `EncoderStage` | class | Max-pool then ConvBlock |
| `DecoderStage` | class | Bilinear upsample, skip concat, ConvBlock |
| `UNet` | class | Parameterised encoder-decoder with a 1x1 head |
| `build_segmentor` | function | Construct a U-Net |

## Inputs and outputs

`stage_widths(base_width, depth) -> tuple[int, ...]`. `stage_widths(64, 4)` equals
`ENCODER_CHANNELS`.

`UNet.forward(x)` maps `(N, C, H, W)` to `(N, 1, H, W)` logits. Height and width
come from the input tensor.

`build_segmentor(in_channels=4, out_channels=1, base_width=64, depth=4,
separable=False) -> UNet`.

## Behavior

1. `base_width` sets the stem channel count. Each encoder stage doubles the width.
2. `depth` sets the stage count including the stem.
3. `separable` replaces each 3x3 convolution with depthwise 3x3 plus pointwise
   1x1.
4. A stem ConvBlock, `depth - 1` encoder stages, a bottleneck, then one decoder
   stage per skip tensor with channel concatenation.
5. A 1x1 convolution emits `out_channels` logits. No sigmoid in the graph.

## Errors and faults

`ValueError` when `base_width` or `depth` is below one.

## Messages

None.

## Configuration

`in_channels` and `out_channels` are constructor arguments. `base_width`
defaults to 64. `depth` defaults to 4. Spatial size is not a constructor field.

## Constraints

This module imports torch at import time. The implementation does not copy
third-party U-Net sources. Registry names use the `unet` family in
[`tools.inference.arch.registry`](registry.md).

## Related documents

- [`tools.inference.arch`](../arch.md)
- [`tools.inference.arch.registry`](registry.md)
- [`tools.inference.train`](../train.md)
