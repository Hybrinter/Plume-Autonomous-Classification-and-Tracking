"""Visual evidence assembly over captured predictions and dataset rows.

Contains:
  - selection: deterministic representative/failure selections.
  - dataset: dataset-row visuals.
  - predictions: prediction-overlay visuals over frozen galleries.

`dataset` and `predictions` render verified captured preview bytes;
`predictions` supports classifier galleries in this phase, with
segmentation deferred to PR14.
"""
