"""Architecture registry tests (resolve is torch-free; build skips without torch)."""

import importlib.util

import pytest
from tools.inference.arch.registry import default_arch, resolve_arch


def test_default_arch() -> None:
    """Defaults are resnet50 and unet."""
    assert default_arch("classifier") == "resnet50"
    assert default_arch("segmentor") == "unet"
    with pytest.raises(ValueError, match="unknown train kind"):
        default_arch("nope")


def test_resolve_arch_fills_empty_and_rejects_unknown() -> None:
    """Empty arch selects the default. Unknown pairs raise."""
    assert resolve_arch("classifier", "") == "resnet50"
    assert resolve_arch("segmentor", "unet") == "unet"
    with pytest.raises(ValueError, match="unknown architecture"):
        resolve_arch("segmentor", "resnet50")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch extra not installed")
def test_build_classifier_and_segmentor() -> None:
    """Registry build returns graphs with the flight I/O ranks."""
    import torch
    from tools.inference.arch.registry import build

    clf = build("classifier", "resnet50", 4).eval()
    seg = build("segmentor", "unet", 4).eval()
    x = torch.zeros(1, 4, 32, 32)
    with torch.no_grad():
        assert clf(x).shape == (1, 1)
        assert seg(x).shape == (1, 1, 32, 32)
