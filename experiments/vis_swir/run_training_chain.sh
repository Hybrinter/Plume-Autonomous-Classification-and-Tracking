#!/usr/bin/env bash
# Train the nine VIS/SWIR seed-0 runs in sequence on one GPU. Each run waits
# for its dataset build to finish (dataset.json present), at most
# PACT_DATASET_WAIT_MIN minutes (default 720); on timeout the run is skipped.
# Logs go beside the runs.
set -u
cd "$(dirname "$0")/../.."
mkdir -p data/processed/runs
wait_cap_min=${PACT_DATASET_WAIT_MIN:-720}
for run in fused_classifier fused_segmentor vis_classifier vis_segmentor swir_classifier \
    swir_segmentor dual_classifier dual_segmentor fused_shufflenet_classifier; do
  cfg="experiments/vis_swir/train_${run}.toml"
  ds=$(sed -n 's/^dataset = "\(.*\)"/\1/p' "$cfg")
  if [ -z "$ds" ]; then
    echo "skip $run: no dataset in $cfg" >&2
    continue
  fi
  waited=0
  if [ ! -f "$ds/dataset.json" ]; then
    echo "waiting for $ds/dataset.json"
    until [ -f "$ds/dataset.json" ] || [ "$waited" -ge "$wait_cap_min" ]; do
      sleep 60
      waited=$((waited + 1))
    done
  fi
  if [ ! -f "$ds/dataset.json" ]; then
    echo "skip $run: no $ds/dataset.json after $wait_cap_min min" >&2
    continue
  fi
  start=$(date +%s)
  uv run python -m tools.ml_models train --config "$cfg" > "data/processed/runs/${run}.log" 2>&1
  echo "$run exit $? min $(( ($(date +%s) - start) / 60 ))"
done
