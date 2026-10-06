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
  - training: strict training records, epoch reduction, and readers.
  - model: model-analysis boundary (unavailable).
  - summaries: tagged versioned summary records.
  - cost: parameter counting and explicitly partial resource evidence.
  - runs: unavailable catalog readers and pure text formatters.
  - pareto: pure frontier/knee helpers and the unavailable reader boundary.
  - metrics: pure metric cores; classifier/calibration/segmentation
    implemented.
  - plots: figure export and dataset figure rendering; the general
    render boundary stays unavailable.
  - visuals: bounded gallery selection and dataset preview rendering.

Import each module by name. This package does not re-export names.
"""
