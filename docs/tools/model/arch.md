# tools.model.arch

**Source:** `packages/tools/src/tools/model/arch/`
**Kind:** package

## Purpose

The arch package holds the segmentor U-Net and the ResNet-50 classifier
builders. Train and export import these modules lazily.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`unet`](arch/unet.md) | module | Clean-room bilinear U-Net, 64-512 channels |
| [`classifier`](arch/classifier.md) | module | torchvision ResNet-50 with a 4-channel stem |

## Package interface

`tools.model.arch.__init__` carries a module docstring only. Callers import
`build_segmentor` and `build_classifier` from the submodules.

## Interactions

Train and export import arch modules inside functions. `import tools.model` does
not import torch.

## Constraints

Submodule import requires the `train` extra (torch and torchvision). The graphs
emit logits. Flight applies sigmoid on the segmentor output.

## Related documents

- [`tools.model`](../model.md)
- [`tools.model.train`](train.md)
- [`tools.model.export`](export.md)
