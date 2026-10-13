"""Parameter counting and unavailable FLOP resource evidence."""

from flight.libs.types import Err
from tools.ml_models.analysis.cost import count_flops, count_params
from tools.ml_models.arch.registry import build


def test_unet_param_count_at_32px() -> None:
    """U-Net at 32 px reports a positive parameter count."""
    model = build("segmentor", "unet", 4).eval()
    assert count_params(model) > 0


def test_count_flops_returns_unavailable() -> None:
    """The removed one-input FLOP executor fails closed."""
    model = build("segmentor", "unet", 4).eval()
    result = count_flops(model, (1, 4, 32, 32))
    assert isinstance(result, Err)
    assert "unavailable" in result.error
