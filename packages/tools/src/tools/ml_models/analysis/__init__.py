"""Evidence-first analysis scaffolds and retained pure helpers.

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
  - model_figure_artifacts: frozen model-figure bundle serialization.
  - prediction_selections: whole-cohort prediction gallery selection.
  - prediction_artifacts: frozen prediction preview/manifest codecs.
  - model: model-analysis boundary (unavailable).
  - summaries: tagged versioned summary records.
  - cost: parameter counting and explicitly partial resource evidence.
  - runs: unavailable catalog readers and pure text formatters.
  - pareto: pure frontier/knee helpers and the unavailable reader boundary.
  - metrics: pure metric cores; classifier/calibration/segmentation/
    spatial/generalization implemented.
  - plots: figure export plus dataset/training/model figure rendering;
    the general render boundary stays unavailable.
  - visuals: bounded gallery selection and dataset/prediction preview
    rendering.

Import each module by name. This package does not re-export names.
"""
