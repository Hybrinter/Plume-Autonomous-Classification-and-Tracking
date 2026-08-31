# tools.inference.arch.blocks

**Source:** `packages/tools/src/tools/inference/arch/blocks.py`
**Kind:** module

## Purpose

This module builds 3x3 convolution stages for the architecture catalog. A dense
stage is one 3x3 convolution. A separable stage is a depthwise 3x3 plus a
pointwise 1x1.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `conv3x3_layers` | function | Convolution layers of one 3x3 stage |
| `conv_norm_relu` | function | Those layers plus a trailing batch-norm and ReLU |

## Inputs and outputs

`conv3x3_layers(in_channels, out_channels, stride=1, dilation=1, separable=False,
mid_norm=False) -> list[nn.Module]`. Bias is off. Padding equals dilation.

`conv_norm_relu(...) -> nn.Sequential`. Appends `BatchNorm2d(out_channels)` and
`ReLU`.

## Behavior

1. A dense stage returns one 3x3 convolution.
2. A separable stage returns a grouped 3x3 then a 1x1 projection.
3. `mid_norm` inserts batch-norm and ReLU between those two convolutions when
   `separable` is set. It has no effect on a dense 3x3.
4. Compact and dilated blocks pass `mid_norm=True` with separable convolutions.
5. The U-Net `ConvBlock` passes `mid_norm=False`.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

This module imports torch. Convolution bias is off. A later batch-norm supplies
the shift.

## Related documents

- [`tools.inference.arch`](../arch.md)
- [`tools.inference.arch.compact`](compact.md)
- [`tools.inference.arch.dilated`](dilated.md)
- [`tools.inference.arch.unet`](unet.md)
