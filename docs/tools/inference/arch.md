# tools.inference.arch

**Source:** `packages/tools/src/tools/inference/arch/`
**Kind:** package

## Purpose

The arch package holds the segmentor U-Net and the ResNet-50 classifier
builders. Train and export import these modules lazily.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`unet`](arch/unet.md) | module | Clean-room bilinear U-Net, 64-512 channels |
| [`classifier`](arch/classifier.md) | module | torchvision ResNet-50 with a 4-channel stem |
| [`registry`](arch/registry.md) | module | Kind plus name to a lazy builder |

## Package interface

`tools.inference.arch.__init__` carries a module docstring only. Callers import
`build_segmentor` and `build_classifier` from the submodules.

## Interactions

Train and export import arch modules to construct networks.

## Constraints

Submodule import uses torch and torchvision from the default tools install. The
graphs emit logits. Flight applies sigmoid on the segmentor output.

## Related documents

- [`tools.inference`](../inference.md)
- [`tools.inference.train`](train.md)
- [`tools.inference.export`](export.md)
