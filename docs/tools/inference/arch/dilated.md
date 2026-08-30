# tools.inference.arch.dilated

**Source:** `packages/tools/src/tools/inference/arch/dilated.py`
**Kind:** module

## Purpose

This module defines a dilated fully-convolutional segmentor family named
`dilatenet`. The stack downsamples twice (or three times when the output stride
is 8), then grows the receptive field with dilated convolutions. There is no
decoder and no skip connections. Output is a full-resolution logit map.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DILATED_PREFIX` | constant | Family prefix string `dilatenet` |
| `DEFAULT_DILATED_WIDTH` | constant | Default stem width (32) |
| `DEFAULT_DILATED_BLOCKS` | constant | Default dilated-block count (4) |
| `DEFAULT_OUTPUT_STRIDE` | constant | Default output stride (4) |
| `VALID_OUTPUT_STRIDES` | constant | Supported output strides `{4, 8}` |
| `DilatedSpec` | class | Parsed width, block count, output stride, and convolution style |
| `dilation_rates` | function | Dilation rate schedule for a block count |
| `DilatedSegmentor` | class | Dilated fully-convolutional segmentor |
| `parse_dilated` | function | Parse a `dilatenet` registry name into a `DilatedSpec` |
| `build_dilated_segmentor` | function | Construct a `DilatedSegmentor` from a spec |

## Inputs and outputs

`dilation_rates(blocks) -> tuple[int, ...]`. Returns `blocks` rates doubling from
one. Example: four blocks give `(1, 2, 4, 8)`. Raises `ValueError` when
`blocks` is below one.

`parse_dilated(name) -> DilatedSpec`. Raises `ValueError` on an unknown family,
modifier token, or unsupported output stride.

`DilatedSegmentor.forward(x)` maps `(N, C, H, W)` to `(N, 1, H, W)` logits.
Height and width come from the input tensor. A bilinear resize restores full
resolution after the head.

`build_dilated_segmentor(spec, in_channels=4, out_channels=1) -> DilatedSegmentor`.

## Behavior

1. Registry names use the `dilatenet` prefix with underscore-separated modifiers.
2. `w<N>` sets the stem width. Default is 32.
3. `d<N>` sets the dilated-block count. Default is 4.
4. `s<N>` sets the output stride held through the body. Supported values are 4
   and 8. Default is 4.
5. `full` selects dense 3x3 convolutions. The default uses depthwise-separable
   convolutions.
6. The stem convolution is always dense. Later stages honour the `separable`
   flag.
7. The body runs at twice the stem width. Two strided blocks reach output
   stride 4; a third strided block is added when the output stride is 8.
8. Each dilated block applies a 3x3 convolution at the scheduled rate without
   further downsampling.
9. A 1x1 head emits `out_channels` logits. No sigmoid in the graph.
10. The head logits are bilinearly resized to the input height and width.
11. Modifiers combine in any order. Example: `dilatenet_w32_d6_s8_full`.

## Errors and faults

`ValueError` on an unknown family, modifier token, unsupported output stride, or
a width, block count, or output stride below the allowed minimum.

## Messages

None.

## Configuration

`in_channels` defaults to 4. `out_channels` defaults to 1. Spatial size is not a
constructor field.

## Constraints

This module imports torch at import time. The graph does not apply sigmoid.
Registry names use the dilated segmentor grammar in
[`tools.inference.arch.registry`](registry.md).

## Related documents

- [`tools.inference.arch`](../arch.md)
- [`tools.inference.arch.unet`](unet.md)
- [`tools.inference.arch.encoder_unet`](encoder_unet.md)
- [`tools.inference.arch.registry`](registry.md)
- [`tools.inference.train`](../train.md)
