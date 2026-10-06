# tools.ml_models.train.losses

**Source:** `packages/tools/src/tools/ml_models/train/losses.py`
**Kind:** module

## Purpose

This module defines the registered training objectives: pixel-wise BCE or
focal terms, a Dice overlap term, and their weighted combinations for the
classifier and the segmentor.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LossName` | type | Registered objective names |
| `LOSS_NAMES` | constant | `bce`, `dice`, `bce_dice`, `focal`, `focal_dice` |
| `LossSpec` | dataclass | Per-name focal/pixel/Dice weight flags |
| `LOSS_SPECS` | constant | `LossSpec` per registered name |
| `DEFAULT_FOCAL_GAMMA` | constant | Focal exponent 2.0 |
| `DEFAULT_FOCAL_ALPHA` | constant | Focal positive weight 0.25 |
| `bce_per_sample` | function | Per-image BCE on logits |
| `dice_per_sample` | function | Per-image soft Dice on logits |
| `focal_per_sample` | function | Per-image focal loss on logits |
| `focal_dice_per_sample` | function | Per-image focal plus Dice |
| `weighted_batch_loss` | function | Batch mean times a source weight |
| `dice_term` | function | Batch mean soft Dice |
| `focal_term` | function | Batch mean focal loss |
| `LossComponents` | dataclass | Per-image total plus each active term |
| `PlumeLoss` | class | Weighted BCE/focal plus Dice module |
| `build_loss` | function | Construct a `PlumeLoss` from a name |

## Inputs and outputs

All losses take raw logits and targets of matching shape: `(N, 1)` for
the classifier and `(N, 1, H, W)` for the segmentor. Per-sample helpers
return `(N,)`; `PlumeLoss.forward` returns a scalar, and
`PlumeLoss.per_sample_components` returns a `LossComponents` of `(N,)`
tensors before the equal-image batch mean.

`build_loss(name, pos_weight=0.0, focal_gamma=2.0, focal_alpha=0.25)`
returns the configured module.

## Behavior

1. `LOSS_SPECS` selects a pixel term (BCE or focal) and an optional Dice
   term; `bce_dice` and `focal_dice` weight both terms at 1.0.
2. `pos_weight` applies only to the BCE pixel term; values at or below
   zero disable it.
3. Dice is computed on sigmoid probabilities with a smoothing constant.
4. `per_sample_components` exposes each configured term per image;
   inactive terms are `None`, `total` is the weighted per-image sum, and
   `forward` is its batch mean. Component values, gradients, and the
   objective weights are identical to the batch reduction.
5. `weighted_batch_loss` multiplies a batch mean by a supplied source
   weight. The training loop reduces `per_sample_components` itself and
   does not call it.

## Errors and faults

`ValueError` from `build_loss` on an unregistered name.

## Messages

None.

## Configuration

`loss`, `pos_weight`, `focal_gamma`, and `focal_alpha` come from
`TrainConfig`.

## Constraints

This module imports torch at import time. Losses consume raw logits; no
sigmoid is baked in.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.train.loop`](loop.md)
