# tools.ml_models.train.losses

**Source:** `packages/tools/src/tools/ml_models/train/losses.py`
**Kind:** module

## Purpose

This module defines training objectives for the classifier and the segmentor.
Every objective consumes raw logits.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LossName` | type alias | Registered objective names |
| `LOSS_NAMES` | constant | Frozen set of registered names |
| `LossSpec` | class | Pixel and Dice weights for one name |
| `LOSS_SPECS` | constant | Name to `LossSpec` table |
| `DEFAULT_FOCAL_GAMMA` | constant | Default focal focusing exponent (2.0) |
| `DEFAULT_FOCAL_ALPHA` | constant | Default focal positive-class weight (0.25) |
| `bce_per_sample` | function | Mean BCE for each image |
| `dice_per_sample` | function | Soft Dice loss for each image |
| `focal_per_sample` | function | Mean focal loss for each image |
| `focal_dice_per_sample` | function | Per-image focal loss plus per-image Dice |
| `weighted_batch_loss` | function | Batch mean multiplied by a source weight |
| `dice_term` | function | Soft Dice loss over a batch |
| `focal_term` | function | Focal loss over a batch |
| `PlumeLoss` | class | Weighted pixel term plus an optional Dice term |
| `build_loss` | function | Construct a `PlumeLoss` from a name |

## Inputs and outputs

`bce_per_sample(logits, targets, pos_weight=0.0) -> Tensor` with shape `(N,)`.

`dice_per_sample(logits, targets) -> Tensor` with shape `(N,)`.

`focal_per_sample(logits, targets, gamma=2.0, alpha=0.25) -> Tensor` with shape
`(N,)`.

`focal_dice_per_sample(logits, targets, gamma=2.0, alpha=0.25) -> Tensor` with
shape `(N,)`.

`weighted_batch_loss(per_sample, source_weight) -> Tensor`. Returns the batch
mean times `source_weight`.

`dice_term(logits, targets) -> Tensor`. Returns scalar `1 - dice` averaged over
the batch.

`focal_term(logits, targets, gamma=2.0, alpha=0.25) -> Tensor`.

`PlumeLoss.forward(logits, targets) -> Tensor`.

`build_loss(name, pos_weight=0.0, focal_gamma=2.0, focal_alpha=0.25) -> PlumeLoss`.
Raises `ValueError` on an unknown name.

## Behavior

1. Registered names: `bce`, `dice`, `bce_dice`, `focal`, `focal_dice`.
2. `LOSS_SPECS` stores `use_focal`, `dice_weight`, and `pixel_weight` for each
   name. `build_loss` looks up that row.
3. `bce` and `focal` use a pixel term only. `dice` uses Dice only.
   `bce_dice` and `focal_dice` sum both terms with weight 1.0 each.
4. `PlumeLoss` selects BCE or focal for the pixel term. A `pos_weight` above
   zero applies to BCE only. Focal uses `focal_alpha` for class balance.
5. Objectives accept segmentor masks `(N, 1, H, W)` and classifier labels
   `(N, 1)`.
6. `bce_per_sample` and `focal_per_sample` average inside each image. A larger
   map keeps the same per-image value. `dice_per_sample` is already one value
   per image. `weighted_batch_loss` multiplies the batch mean by the source
   weight.

## Errors and faults

`ValueError` when `build_loss` receives an unknown name. `ValueError` when
`weighted_batch_loss` receives a source weight that is not finite and above
zero.

## Messages

None.

## Configuration

`DEFAULT_FOCAL_GAMMA` is 2.0. `DEFAULT_FOCAL_ALPHA` is 0.25. `pos_weight` at
or below zero disables BCE positive-class weighting.

## Constraints

This module imports torch. Loss values feed the train loop and are not published
on the bus.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.train.loop`](loop.md)
