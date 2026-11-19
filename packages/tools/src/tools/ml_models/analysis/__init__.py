"""Evidence-first measurement, freezing, and rendering modules.

Contains:
  - contracts: sample-key, metric, and split-evidence records.
  - config: analysis, evaluation, and plot configuration records.
  - artifacts: versioned codecs, typed tables, and safe publication.
  - capture: bounded prediction/evidence sink protocol.
  - evaluate: identity-bound split evaluation with canonical train
    populations.
  - dataset: whole-dataset measurement and summary assembly.
  - dataset_artifacts: frozen-measurement bundle persistence.
  - dataset_figures: frozen dataset figure-coordinate recipes.
  - dataset_previews: bounded exact preview capture.
  - dataset_render: figure/preview orchestration and bundle publication.
  - generalization_artifacts: frozen generalization-evidence bundle
    codecs.
  - training: strict training records, epoch reduction, and readers.
  - training_figures: frozen training-history figure-coordinate recipes.
  - training_artifacts: frozen recipe bundle serialization.
  - model_figures: frozen model-chart recipes and scalar reductions.
  - classifier_figures: frozen classifier figure inventory.
  - segmentation_figures: frozen segmentor extent/localization figure
    inventory.
  - generalization_figures: frozen stratum/baseline/heatmap figure
    copies.
  - model_figure_artifacts: frozen model-figure bundle serialization.
  - prediction_display: verified segmentor display arrays from cached
    logits.
  - prediction_selections: whole-cohort prediction gallery selection.
  - prediction_artifacts: frozen prediction preview/manifest codecs.
  - model_inputs: verified run/checkpoint/dataset input loading.
  - model_measurement: selected-checkpoint measurement policy.
  - model_summary: canonical model/training summary assembly.
  - model_artifacts: frozen evidence bundle assembly and tables.
  - model_render: frozen-recipe rendering and render-only republication.
  - model: the measure-freeze-render-publish boundary.
  - summaries: tagged versioned summary records.
  - cost: parameter counting and explicitly partial resource evidence.
  - runs: unavailable catalog readers and pure text formatters.
  - pareto: pure frontier/knee helpers and the unavailable reader boundary.
  - metrics: pure metric cores; classifier/calibration/segmentation/
    spatial/generalization implemented.
  - plots: figure export plus dataset/training/model figure rendering
    and the render-only bundle boundary.
  - visuals: bounded gallery selection and dataset/prediction preview
    rendering.

Import each module by name. This package does not re-export names.
"""
