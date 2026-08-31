# tools.inference.arch

**Source:** `packages/tools/src/tools/inference/arch/`
**Kind:** package

## Purpose

The arch package holds segmentor and classifier network builders. Architecture
names form a grammar resolved by the registry. Train and export import these
modules.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`grammar`](arch/grammar.md) | module | Shared architecture-name modifier tokens |
| [`blocks`](arch/blocks.md) | module | Shared 3x3 dense and separable convolutions |
| [`unet`](arch/unet.md) | module | Parameterised scratch U-Net segmentor |
| [`encoder_unet`](arch/encoder_unet.md) | module | ResNet-encoder U-Net segmentor |
| [`dilated`](arch/dilated.md) | module | Dilated fully-convolutional `dilatenet` segmentor |
| [`classifier`](arch/classifier.md) | module | torchvision backbones with a 4-channel stem |
| [`compact`](arch/compact.md) | module | Compact `pactnet` classifier family |
| [`stem`](arch/stem.md) | module | Band-count surgery for pretrained stems |
| [`registry`](arch/registry.md) | module | Kind plus grammar name to a builder |

## Package interface

`tools.inference.arch.__init__` carries a module docstring only. Callers import
builders from the submodules or call `registry.build`.

## Interactions

Train and export import arch modules to construct networks. The stem module
supports pretrained weight transfer for classifiers and encoder segmentors.

## Constraints

Submodule import uses torch and torchvision from the default tools install. The
graphs emit logits. Flight applies sigmoid on the segmentor output.

## Related documents

- [`tools.inference`](../inference.md)
- [`tools.inference.train`](train.md)
- [`tools.inference.export`](export.md)
