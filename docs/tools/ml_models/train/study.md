# tools.ml_models.train.study

**Source:** `packages/tools/src/tools/ml_models/train/study.py`
**Kind:** module

## Purpose

This module trains a classifier or a segmentor with one optimizer, one loss,
and one early-stopping rule.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TrainConfig` | class | Epochs, learning rate, patience, seed |
| `EpochLoss` | class | Train, validation, and test loss for one epoch |
| `TrainResult` | class | Best epoch, validation loss, weights, and history |
| `run_training` | function | The shared loop |

## Inputs and outputs

`run_training(model, train_loader, val_loader, config, target) -> TrainResult`.

``target`` is ``label`` for the classifier and ``mask`` for the segmentor.

## Behavior

1. Each train batch is flipped on either axis and rotated by a multiple of 90
   degrees. The mask receives the same transform.
2. The loss is binary cross-entropy with logits. The optimizer is AdamW. The
   learning rate follows a cosine schedule over ``epochs``.
3. Validation loss is computed without augmentation. An optional test loader
   is scored the same way and stored on the history. Training stops after
   ``patience`` epochs without a new minimum validation loss. The returned
   weights are that minimum. History epochs are one-based.
4. A batch may include an annotation flag. The segmentor omits tiles whose
   flag is 0. The classifier uses every tile. The cosine schedule advances on
   an epoch that updates the weights.

## Errors and faults

`ValueError` when ``target`` is unknown, a loader yields no batches, a
batch does not hold 3 or 4 tensors, or validation keeps no samples.

## Messages

None.

## Configuration

Defaults are 40 epochs, learning rate ``1e-3``, weight decay ``1e-4``, and
patience 8.

## Constraints

The same loop serves both heads. The target argument selects the tensor.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.train.tiles`](tiles.md)
- [`tools.ml_models.analysis.scores`](../analysis/scores.md)
