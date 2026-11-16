#!/usr/bin/env bash
# Reproduce the VIS/SWIR fusion-trade results in experiments/vis_swir/results.
#
# Stage "seeds" trains the extra classifier seeds (about 30 min each). Stage
# "score" runs inference only (about 1 min per run, about 25 min for the
# late-fusion searches). Stage "thresholds" needs the seed-0 model analyses
# from run_analysis_chain.sh. Run one stage at a time and never during
# another training or analysis job: the machine runs out of commit memory.
#
#   bash experiments/vis_swir/run_fusion_results.sh seeds|score|thresholds|plots
set -eu
cd "$(dirname "$0")/../.."
P=data/processed
R=experiments/vis_swir/results
CFG=experiments/vis_swir/analyze_model.toml
PAIRED=$P/paired_eval_fixed
fusion() { uv run python -m tools.ml_models.fusion "$@"; }

case "${1:-}" in
seeds)
  for s in 1 2; do
    for base in fused_classifier vis_classifier; do
      uv run python -m tools.ml_models train --config "experiments/vis_swir/train_${base}.toml" \
        --seed "$s" --run-id "${base}_s${s}" > "$P/runs/${base}_s${s}.log" 2>&1
    done
  done
  for s in 0 1 2; do
    uv run python -m tools.ml_models train \
      --config experiments/vis_swir/train_dual_nodrop_classifier.toml \
      --seed "$s" --run-id "dual_nodrop_classifier_s${s}" \
      > "$P/runs/dual_nodrop_classifier_s${s}.log" 2>&1
  done
  ;;
score)
  # Paired-input runs score on the fixed paired build; VIS-only on its own build.
  for run in fused_classifier fused_classifier_s1 fused_classifier_s2 \
             dual_nodrop_classifier_s0 dual_nodrop_classifier_s1 dual_nodrop_classifier_s2 \
             dual_classifier fused_shufflenet_classifier fused_segmentor dual_segmentor; do
    fusion test --config $CFG --run "$P/runs/$run" --dataset $PAIRED --out "$R/test/$run.json"
  done
  for run in vis_classifier vis_classifier_s1 vis_classifier_s2 vis_segmentor; do
    fusion test --config $CFG --run "$P/runs/$run" --out "$R/test/$run.json"
  done
  for run in swir_classifier swir_segmentor; do
    fusion test --config $CFG --run "$P/runs/$run" --dataset $P/swir_eval_fixed \
      --out "$R/test/$run.json"
  done
  for vis in vis_classifier vis_classifier_s1 vis_classifier_s2; do
    fusion late --config $CFG --vis-run "$P/runs/$vis" --swir-run $P/runs/swir_classifier \
      --reference-run $P/runs/fused_classifier --dataset $PAIRED \
      --metric average_precision --out "$R/late/$vis.json"
  done
  fusion late --config $CFG --vis-run $P/runs/vis_segmentor --swir-run $P/runs/swir_segmentor \
    --reference-run $P/runs/fused_segmentor --dataset $PAIRED \
    --metric foreground_iou_mean_positive_images --out "$R/late/vis_segmentor.json"
  ;;
thresholds)
  for run in fused_classifier dual_classifier vis_classifier swir_classifier; do
    fusion thresholds --analysis "$P/${run}_analysis" --out "$R/thresholds/$run.json" \
      --plot "$R/thresholds/$run.png" --title "${run//_/ }"
  done
  ;;
plots)
  T=$R/test
  fusion seed-plot --out "$R/seed_ap.png" --title "Classifier test AP per training seed" \
    --entry "Early=$T/fused_classifier.json,$T/fused_classifier_s1.json,$T/fused_classifier_s2.json" \
    --entry "Mid=$T/dual_nodrop_classifier_s0.json,$T/dual_nodrop_classifier_s1.json,$T/dual_nodrop_classifier_s2.json" \
    --entry "Late=$R/late/vis_classifier.json,$R/late/vis_classifier_s1.json,$R/late/vis_classifier_s2.json" \
    --entry "VIS only=$T/vis_classifier.json,$T/vis_classifier_s1.json,$T/vis_classifier_s2.json" \
    --entry "Early, ShuffleNet=$T/fused_shufflenet_classifier.json"
  ;;
*)
  echo "usage: $0 seeds|score|thresholds|plots" >&2
  exit 2
  ;;
esac
