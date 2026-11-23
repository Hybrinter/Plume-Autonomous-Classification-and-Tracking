"""Two-input ONNX export, validation, acceptance, and pair gating for GSD-
conditioned flight models.

Exports ``pactnet``/``dilatenet`` checkpoints as dynamic-batch (image, gsd)
ONNX graphs with JSON sidecars (``ModelManifest``), validates artifact bytes
and declared shapes, runs finished-dataset acceptance evaluation, and emits
classifier/segmentor pair manifests only for accepted artifacts. Importing
this package never imports onnx/onnxruntime or torch.
"""

__all__: list[str] = []
