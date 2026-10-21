#!/usr/bin/env bash
# Analyze each trade-study run once its training execution is COMPLETED.
# Fused and SWIR-only runs evaluate on the fixed-mix builds.
set -u
cd "$(dirname "$0")/../.."
for job in fused_classifier:paired_eval_fixed fused_segmentor:paired_eval_fixed \
    vis_classifier: vis_segmentor: \
    swir_classifier:swir_eval_fixed swir_segmentor:swir_eval_fixed; do
  run=${job%%:*}; eval_ds=${job#*:}
  out="data/processed/${run}_analysis"
  [ -f "$out/summary.json" ] && continue
  until uv run python -c "import json,sys; sys.exit(json.load(open(sys.argv[1]))['status'] != 'COMPLETED')"       "data/processed/runs/$run/execution.json" 2>/dev/null; do
    sleep 60
  done
  extra=()
  [ -n "$eval_ds" ] && extra=(--dataset "data/processed/$eval_ds")
  start=$(date +%s)
  uv run python -m tools.ml_models analyze --config experiments/vis_swir/analyze_model.toml \
    --run "data/processed/runs/$run" --out "$out" "${extra[@]}" \
    > "data/processed/${run}_analyze.log" 2>&1
  echo "$run exit $? min $(( ($(date +%s) - start) / 60 ))"
done
