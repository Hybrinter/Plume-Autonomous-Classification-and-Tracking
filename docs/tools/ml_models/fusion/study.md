# tools.ml_models.fusion.study

**Source:** `packages/tools/src/tools/ml_models/fusion/study.py`
**Kind:** module

## Purpose

This module scores trained runs on one split and searches the late-fusion VIS
weight.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LateFusionResult` | dataclass | Chosen VIS weight, validation scores, and test metrics |
| `load_run` | function | Verified model inputs for one run on one dataset |
| `score_split` | function | Aggregate metrics of one model on one split |
| `weight_key` | function | Two-decimal text key of one VIS weight in the output JSON |
| `check_weights` | function | Nonempty VIS weight grid in [0, 1] with distinct keys |
| `check_gsd_reference` | function | One GSD reference for both branches and the dataset |
| `check_bands` | function | VIS then SWIR bands equal to the paired dataset bands |
| `late_fusion` | function | VIS weight on validation, then test once |

## Inputs and outputs

`load_run(base, run, dataset) -> Result[ModelInputs, str]`.

`score_split(model, inputs, base, split) -> Result[dict[str, float], str]`.

`weight_key(weight) -> str`.

`check_weights(weights) -> Result[None, str]`.

`check_gsd_reference(vis_m, swir_m, paired_m) -> Result[None, str]`.

`check_bands(vis_bands, swir_bands, paired_bands) -> Result[None, str]`.

`late_fusion(vis, swir, paired, base, metric, weights) -> Result[LateFusionResult, str]`.

## Behavior

1. `load_run` replaces `run`, `out`, and `dataset` in the analysis
   configuration and calls `tools.ml_models.analysis.model_inputs.load_model_inputs`.
   It does not write the `out` directory.
2. `score_split` calls `tools.ml_models.analysis.evaluate.evaluate_split` with
   no capture. The result holds every scalar metric that has a float value.
3. `late_fusion` scores the validation split at each weight, selects a weight
   with `late.select_weight`, and scores the test split once at that weight.
   The `paired` inputs supply the evaluation dataset and the task. Only the
   models of the `vis` and `swir` inputs are used.
4. The model GSD input is `ln(gsd / reference)`. The evaluator encodes it
   with the paired dataset reference, and both branches get the same tensor.
   `late_fusion` calls `check_gsd_reference` with the training-dataset
   reference of each branch and the paired evaluation reference.
5. Before any inference, `late_fusion` calls `check_weights`, then
   `check_gsd_reference`, then `check_bands`. `check_weights` accepts a
   nonempty grid of weights in [0, 1] whose `weight_key` values are distinct.
   `weight_key` formats a weight with two decimals. The command line uses it for
   the keys of the validation score per weight.
6. `check_bands` accepts the VIS training bands followed by the SWIR training
   bands when they equal the paired evaluation bands in order. The VIS branch
   reads the first planes, one plane for each VIS training band. The SWIR
   branch reads the other planes.

## Errors and faults

Each function returns `Err` with the load or evaluation error. `late_fusion`
returns `Err` before any inference when:

- the weight grid is empty, a weight is outside [0, 1] or is NaN, or two
  weights have the same two-decimal key;
- a branch training reference differs from the paired evaluation reference;
- the VIS bands followed by the SWIR bands differ from the paired evaluation
  bands. The message gives the three band tuples.

`late_fusion` also returns `Err` when the validation metric is not available.

## Messages

None.

## Configuration

The analysis configuration supplies the batch size, device, and score settings.

## Constraints

Inference runs on the configured device.

## Related documents

- [`tools.ml_models.fusion`](../fusion.md)
- [`tools.ml_models.fusion.late`](late.md)
- [`tools.ml_models.fusion.cli`](cli.md)
