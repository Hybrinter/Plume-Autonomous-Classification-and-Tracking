#!/usr/bin/env bash
# Analyze each trade-study run once its training execution is COMPLETED.
# Fused, dual, and SWIR-only runs evaluate on the fixed-mix builds. A run whose
# status is FAILED or INTERRUPTED is skipped. A missing or unreadable
# execution.json is polled for at most PACT_RUN_WAIT_MIN minutes (default 60),
# then the run is skipped.
set -u
cd "$(dirname "$0")/../.."
wait_cap_min=${PACT_RUN_WAIT_MIN:-60}
for job in fused_classifier:paired_eval_fixed fused_segmentor:paired_eval_fixed \
    vis_classifier: vis_segmentor: \
    swir_classifier:swir_eval_fixed swir_segmentor:swir_eval_fixed \
    dual_classifier:paired_eval_fixed dual_segmentor:paired_eval_fixed; do
  run=${job%%:*}; eval_ds=${job#*:}
  out="data/processed/${run}_analysis"
  [ -f "$out/summary.json" ] && continue
  exe="data/processed/runs/$run/execution.json"
  polls=0; missing=0
  while :; do
    status=ABSENT
    [ -f "$exe" ] && status=$(uv run python -c \
      "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$exe")
    case "$status" in
      COMPLETED) break ;;
      FAILED | INTERRUPTED)
        echo "skip $run: status $status" >&2
        continue 2
        ;;
      RUNNING) ;;
      *)
        if [ "$missing" -ge "$wait_cap_min" ]; then
          echo "skip $run: no readable $exe after $wait_cap_min min" >&2
          continue 2
        fi
        missing=$((missing + 1))
        ;;
    esac
    [ "$polls" -eq 0 ] && echo "waiting for $run"
    polls=$((polls + 1))
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
