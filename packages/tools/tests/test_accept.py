"""Compatibility import for the moved acceptance gate."""

from tools.accept import (
    accept_artifact,
    accept_classifier_artifact,
    compute_iou,
    load_manifest,
)


def test_tools_accept_reexports_model_accept() -> None:
    """tools.accept still exposes the gate after the move into tools.model."""
    assert callable(accept_artifact)
    assert callable(accept_classifier_artifact)
    assert callable(compute_iou)
    assert callable(load_manifest)
