#!/usr/bin/env bash
# Train the VIS/SWIR matrix in sequence on one GPU. Each run waits for its
# dataset build to finish (dataset.json present). Logs go beside the runs.
set -u
cd "$(dirname "$0")/../.."
for run in fused_classifier fused_segmentor vis_classifier vis_segmentor swir_classifier swir_segmentor; do
  cfg="experiments/vis_swir/train_${run}.toml"
  ds=$(sed -n 's/^dataset = "\(.*\)"/\1/p' "$cfg")
  until [ -f "$ds/dataset.json" ]; do sleep 60; done
  start=$(date +%s)
  uv run python -m tools.ml_models train --config "$cfg" > "data/processed/runs/${run}.log" 2>&1
  echo "$run exit $? min $(( ($(date +%s) - start) / 60 ))"
done
