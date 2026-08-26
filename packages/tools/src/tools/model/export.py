"""ONNX export scaffold for tools.model.

This module is a stub. Torch-to-ONNX export and promote-into-data/models/ land
in a later layer. Importing this module does not import torch.

Contains:
  - export: raises NotImplementedError until the export layer lands.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations


def export() -> None:
    """Export trained weights to frozen ONNX artifacts.

    Returns:
        None. This scaffold always raises.

    Raises:
        NotImplementedError: Export is not in this layer.
    """
    raise NotImplementedError("tools.model.export is a scaffold; ONNX export is not in this layer")
