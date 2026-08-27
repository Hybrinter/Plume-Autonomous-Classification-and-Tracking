# tools.model.arch.unet

**Source:** `packages/tools/src/tools/model/arch/unet.py`
**Kind:** module

## Purpose

This module defines a clean-room encoder-decoder segmentor. Channel widths are
64, 128, 256, and 512. Upsample is bilinear. Output is a logit map.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ENCODER_CHANNELS` | constant | `(64, 128, 256, 512)` |
| `ConvBlock` | class | Two 3x3 conv-BN-ReLU stages |
| `EncoderStage` | class | Max-pool then ConvBlock |
| `DecoderStage` | class | Bilinear upsample, skip concat, ConvBlock |
| `UNet` | class | Four-level encoder-decoder with a 1x1 head |
| `build_segmentor` | function | Construct a U-Net |

## Inputs and outputs

`UNet.forward(x)` maps `(N, C, H, W)` to `(N, 1, H, W)` logits. Height and width
come from the input tensor.

`build_segmentor(in_channels=4, out_channels=1) -> UNet`.

## Behavior

1. Stem ConvBlock at 64 channels.
2. Three encoder stages to 512 channels, then a 512-channel bottleneck.
3. Four decoder stages with skip concatenation.
4. 1x1 convolution to `out_channels` logits. No sigmoid in the graph.

## Errors and faults

None beyond torch runtime errors.

## Messages

None.

## Configuration

`in_channels` and `out_channels` are constructor arguments. Spatial size is not
a constructor field.

## Constraints

This module imports torch at import time. Call it only from train or export
after the train extra is installed. The implementation does not copy third-party
U-Net sources.

## Related documents

- [`tools.model.arch`](../arch.md)
- [`tools.model.train`](../train.md)
