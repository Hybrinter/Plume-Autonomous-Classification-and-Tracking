# tools.ml_models.fusion.late

**Source:** `packages/tools/src/tools/ml_models/fusion/late.py`
**Kind:** module

## Purpose

This module fuses a separately trained VIS model and SWIR model by a weighted
sum of their logits.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DEFAULT_VIS_WEIGHTS` | constant | VIS weights 0 to 1 in 0.1 steps |
| `LateFusion` | class | Weighted logit sum on one paired input |
| `select_weight` | function | VIS weight with the best validation score |

## Inputs and outputs

`LateFusion(vis, swir, vis_weight=0.5, vis_planes=3)`;
`forward(image, gsd) -> Tensor`; `set_vis_weight(vis_weight) -> None`.

`select_weight(val_scores) -> float`.

## Behavior

1. The paired input holds BLUE, GREEN, RED, SWIR in that plane order.
2. The VIS model reads the first `vis_planes` planes. The SWIR model reads the
   remaining planes. Both models get the same GSD encoding.
3. The output is `w * vis + (1 - w) * swir`. Weight 1 is VIS only. Weight 0 is
   SWIR only.
4. `select_weight` returns the weight with the highest score. A tie keeps the
   larger VIS weight.

## Errors and faults

`LateFusion` raises `ValueError` for a weight outside 0 to 1 or for
`vis_planes` less than 1. `select_weight` raises `ValueError` on an empty mapping.

## Messages

None.

## Configuration

None.

## Constraints

The module does not load checkpoints.

## Related documents

- [`tools.ml_models.fusion`](../fusion.md)
- [`tools.ml_models.fusion.study`](study.md)
