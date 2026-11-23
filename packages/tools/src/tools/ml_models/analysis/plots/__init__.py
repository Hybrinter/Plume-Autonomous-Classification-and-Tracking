"""Figure rendering over captured evidence.

Plotting consumes frozen measurement evidence and never reruns inference.
This package replaces the removed CSV-history report figures; the legacy
``history_figures``/``overlay_figures`` APIs are gone.

Contains:
  - common: figure/export conventions and the render boundary.
  - dataset: dataset-analysis figures.
  - training: training-history figures.
  - model: shared model-chart recipe renderer.
  - classifier: classifier family binding over the model renderer.
  - segmentation: segmentation-evidence figures.
  - generalization: generalization-evidence figures.

`common` supplies the export boundary and the unavailable general
render entry point; `dataset` and `training` render their frozen
recipes, while `model` and the `classifier`/`segmentation`/
`generalization` family bindings render frozen model-chart recipes.
"""
