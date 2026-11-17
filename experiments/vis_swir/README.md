# VIS/SWIR trade-study builds

Configs for the paired Zenodo builds (`--source zenodo-paired`). Every build
uses the VIS table `data/calibration/weights/imx267_placeholder_v0.toml`, the
default 5-45 deg elevation bins, and the default seed-0 group split, so all
builds share the same test scenes.

| Config | Bands | SWIR mix | Spec | Role |
| --- | --- | --- | --- | --- |
| `paired_train_randomized.toml` | BLUE, GREEN, RED, SWIR | randomized | default | fused training |
| `paired_eval_fixed.toml` | BLUE, GREEN, RED, SWIR | fixed | `build_eval_spec.toml` | fused evaluation |
| `vis_only.toml` | BLUE, GREEN, RED | n/a | default | VIS-only training and evaluation |
| `swir_only_train_randomized.toml` | SWIR | randomized | default | SWIR-only training |
| `swir_only_eval_fixed.toml` | SWIR | fixed | `build_eval_spec.toml` | SWIR-only evaluation |

Example:

    uv run python -m tools.ml_models dataset build --source zenodo-paired \
      --out data/processed/paired_train_rand \
      --images-tar data/raw/images.tar.gz \
      --labels-tar data/raw/segmentation_labels.tar.gz \
      --weights-path data/calibration/weights/imx267_placeholder_v0.toml \
      --paired-config experiments/vis_swir/paired_train_randomized.toml

## Training

`train_{fused,vis,swir}_{classifier,segmentor}.toml` train one model per
input set with one shared recipe: SGD, learning rate 0.01, cosine schedule,
seed 0. Classifiers run 12 epochs with patience 4; segmentors run 40 epochs
with patience 10. Fused and SWIR-only models train on the randomized builds.

`run_training_chain.sh` runs the six configs in order. Before each run it
waits for `dataset.json` in the configured dataset, so it can start while
the builds are still in progress. Each run writes a log next to its run
directory.

`analyze_model.toml` holds the model-analysis settings: final test on, GPU,
batch 64, and a 2 GiB capture budget. The default 256 MiB budget is too small
for the full validation and test splits. Evaluate fused and SWIR-only runs on
the fixed builds:

    uv run python -m tools.ml_models analyze \
      --config experiments/vis_swir/analyze_model.toml \
      --run data/processed/runs/fused_classifier \
      --out data/processed/fused_classifier_analysis \
      --dataset data/processed/paired_eval_fixed

`run_analysis_chain.sh` analyzes each run after its training status is
`COMPLETED` and skips runs that already have an analysis. One analysis is
about 1.2 GB.

Camera values are estimates. See the paired source documentation.

## Fusion-trade results

`run_fusion_results.sh` reproduces the files in `results/`. Run one stage at a
time, and never during another training or analysis job:

    bash experiments/vis_swir/run_fusion_results.sh seeds       # extra seeds, about 30 min each
    bash experiments/vis_swir/run_fusion_results.sh score       # test scores and late fusion
    bash experiments/vis_swir/run_fusion_results.sh thresholds  # needs the seed-0 analyses
    bash experiments/vis_swir/run_fusion_results.sh plots

The `seeds` stage uses `train --seed N --run-id NAME` on the seed-0 configs.
The `score` stage skips a run that has no `checkpoints/best.pt`. The tools are
in `tools.ml_models.fusion` (`python -m tools.ml_models.fusion --help`).

Options:

- Early: one network on the 4-band BLUE, GREEN, RED, SWIR input (`fused_*`).
- Mid: two branches joined after the stem (`dual_*`). `dual_nodrop` has no
  modality dropout.
- Late: the VIS and SWIR single-camera models, with the logits mixed as
  `w * vis + (1 - w) * swir`. The validation split sets `w` on a 0 to 1 grid
  in steps of 0.1.

All scores are on the test split of the fixed builds (15240 tiles).

### Classifier, three training seeds

Test average precision (AP). `results/seed_ap.png` shows each seed.

| Option | Seed 0 | Seed 1 | Seed 2 | Mean |
| --- | --- | --- | --- | --- |
| Early | 0.756 | 0.738 | 0.758 | 0.751 |
| Mid, no dropout | 0.725 | 0.747 | 0.753 | 0.741 |
| Late | 0.752 | 0.759 | 0.753 | 0.755 |
| VIS only | 0.728 | 0.721 | 0.720 | 0.723 |
| SWIR only | 0.652 | - | - | - |
| Mid, dropout | 0.725 | - | - | - |

The late-fusion weight is `w = 0.5` for all three VIS seeds
(`results/late/vis_classifier*.json` holds the validation AP for each `w`).
The seed spread of one option (up to 0.028 AP) is as large as the difference
between Early, Mid, and Late. Early and Late are better than VIS only on all
three seeds. Mid seed 0 is 0.003 AP below VIS seed 0.

### Segmentor

Test foreground IoU, mean over the positive images, seed 0.

| Option | IoU |
| --- | --- |
| Early | 0.243 |
| Mid | 0.248 |
| VIS only | 0.249 |
| Late (`w = 1.0`, VIS only) | 0.246 |
| SWIR only | 0.170 |

The validation split sets the late-fusion weight to 1.0: the SWIR segmentor
adds no IoU. The late value is on the paired build and the VIS-only value is
on the VIS build, which gives the small difference.

### Classifier thresholds per elevation

`results/thresholds/` holds the test F1 at three logit thresholds: fixed
(logit 0), one global threshold set on the validation split, and one
threshold per elevation bin set on the validation split.

| Model | Fixed | Global | Per elevation | 45 deg, fixed | 45 deg, per elevation |
| --- | --- | --- | --- | --- | --- |
| Early | 0.634 | 0.680 | 0.688 | 0.388 | 0.626 |
| Mid, dropout | 0.583 | 0.646 | 0.662 | - | - |
| VIS only | 0.651 | 0.662 | 0.664 | - | - |
| SWIR only | 0.527 | 0.548 | 0.568 | - | - |

### Compute per tile

Counted Conv2d and Linear operations (2 per multiply-add) for one tile of the
largest bin, 39 x 40 px. The count is a lower bound, not a latency.

| Option | Classifier MFLOP | Segmentor MFLOP | Classifier parameters |
| --- | --- | --- | --- |
| Early | 1.69 | 5.15 | 42577 |
| Mid | 1.98 | 6.39 | 44689 |
| Late (VIS + SWIR) | 2.91 | 9.37 | 84578 |
| VIS only | 1.57 | 4.92 | 42433 |
