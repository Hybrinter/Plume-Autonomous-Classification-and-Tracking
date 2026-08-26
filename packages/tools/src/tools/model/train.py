"""Train loop scaffold for tools.model.

This module is a stub. The plain-torch train loop lands in a later layer.
Importing this module does not import torch.

Contains:
  - train: raises NotImplementedError until the train layer lands.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations


def train() -> None:
    """Run model training.

    Returns:
        None. This scaffold always raises.

    Raises:
        NotImplementedError: The train loop is not in this layer.
    """
    raise NotImplementedError(
        "tools.model.train is a scaffold; the train loop is not in this layer"
    )
