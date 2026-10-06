"""Figure rendering over captured evidence.

Plotting consumes frozen measurement evidence and never reruns inference.
This package replaces the removed CSV-history report figures; the legacy
``history_figures``/``overlay_figures`` APIs are gone.

Contains:
  - common: figure/export conventions and the render boundary.
  - dataset: dataset-analysis figures.
  - training: training-history figures.
  - classifier: classifier-evidence figures.
  - segmentation: segmentation-evidence figures.
  - generalization: generalization-evidence figures.

`common` supplies the export boundary and the unavailable general
render entry point; `dataset` and `training` are implemented renderers
over their frozen recipes; `classifier`, `segmentation`, and
`generalization` remain scaffolds.
"""
