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

Evaluate fused and SWIR-only runs on the fixed builds:

    uv run python -m tools.ml_models analyze \
      --run data/processed/runs/fused_classifier \
      --out data/processed/fused_classifier_analysis \
      --dataset data/processed/paired_eval_fixed --final-test

Camera values are estimates. See the paired source documentation.
