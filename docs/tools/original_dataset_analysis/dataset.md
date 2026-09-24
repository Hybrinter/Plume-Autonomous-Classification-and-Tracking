# tools.original_dataset_analysis.dataset

**Source:** `packages/tools/src/tools/original_dataset_analysis/dataset.py`
**Kind:** module

## Purpose

This module yields one normalized tile, a presence label, and a mask at a
chosen band subset and legal side.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TileReader` | type | Callable that returns a native stack |
| `StudyDataset` | class | Torch dataset of image, label, and mask |

## Inputs and outputs

`StudyDataset.__getitem__(index) -> tuple[Tensor, Tensor, Tensor]`.

The image is ``(C, side, side)``. The label is ``(1,)``. The mask is
``(1, side, side)``.

## Behavior

1. The reader returns a native ``(bands, 120, 120)`` stack in file order.
2. The subset indices select channels. Those channels are coarsened, then scaled
   with frozen moments.
3. A tile with ``polygons is None`` uses an empty polygon list.

## Errors and faults

`ValueError` when the moment width does not match the subset, or when the
reader returns a stack the grid refuses.

## Messages

None.

## Configuration

``mask_rule`` defaults to ``half``.

## Constraints

Moments are an input. The dataset does not refit them.

## Related documents

- [`tools.original_dataset_analysis`](../original_dataset_analysis.md)
- [`tools.original_dataset_analysis.models`](models.md)
