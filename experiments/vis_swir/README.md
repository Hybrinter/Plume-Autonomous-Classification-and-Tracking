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

Camera values are estimates. See the paired source documentation.
